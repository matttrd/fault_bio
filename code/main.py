import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchnet.meter import ClassErrorMeter, ConfusionMeter, TimeMeter, AverageValueMeter
from loader import data_ingredient, load_data 
from sacred import Experiment
import threading
from hook import *
from deep_anomaly_model import ContextNet, BasicTaskNet

# local thread used as a global context
ctx = threading.local()
ex = Experiment('basic_tasks', ingredients=[data_ingredient])
ctx.ex = ex


@ctx.ex.config
def cfg():
    '''
    Base configuration
    '''
    # architecture
    arch = 'allcnn'
    # number of data loading workers (default: 4)
    j = 1
    # number of total B to run
    epochs = 200
    # starting epoch
    start_epoch = 0
    # batch size
    b = 128
    # initial learning rate
    lr = 0.1
    # momentum
    momentum = 0.9
    # weight decay (ell-2)
    wd = 0.
    # print-freq
    print_freq = 50
    # path to latest checkpoint
    resume = ''
    # evaluate a model on validation set
    evaluate = False
    # if True, load a pretrained model
    pretrained = False
    # seed
    seed = None
    # gpu index (or starting gpu idx)
    g = int(0)
    # number of gpus for data parallel
    ng = int(0)
    # learning schedule
    lrs = '[[0,0.1],[60,0.02],[120,0.004],[160,0.0008],[180,0.0001]]'
    # dropout (if any)
    d = 0.
    # file logger
    fl = False
    # Tensorflow logger
    tfl = False
    dbl = False
    # output dir
    o = '../results/'
    #save model
    save = False
    #whitelist for filename
    whitelist = '[]'
    # marker
    marker = ''
    stride = 50
    win_past_len = 200
    win_pred_len = 40
    ctx_ncls = [5,2,2]
    ctx_ksz = 2
    num_ncls = [10,20,30]
    ksz = 2

best_top1 = 0

# for some reason, the db must me created in the global scope
if ex.configurations[0]()['dbl']:
    from sacred.observers import MongoObserver
    from sacred.utils import apply_backspaces_and_linefeeds
    print('Creating database')
    ctx.ex.observers.append(MongoObserver.create())
    ctx.ex.captured_out_filter = apply_backspaces_and_linefeeds

@data_ingredient.capture
def init(name):
    ctx.opt = init_opt(ctx)
    ctx.opt['dataset'] = name
    ctx.metrics = dict()
    ctx.metrics['best_top1'] = best_top1
    ctx.hooks = None
    register_hooks(ctx)
    

@batch_hook(ctx, mode='train')
def runner(inputs, targets, model, criterion, optimizer):
    yh_cl, yh_reg = basik_tasks_model(inputs[0], inputs[1])
    
    loss = criterion[0](yh_cl, targets[0])
    ctx.losses['cl'].add(loss.item())
    loss += opt['lambda'] * criterion[1](output, targets[1])
    # measure accuracy and record loss
    ctx.errors.add(yh_cl.data, targets[0].data)
    ctx.losses['reg'].add(loss.item() - ctx.losses['cl'].value())
    ctx.losses['tot'].add(loss.item())
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_stats = {
             'loss_cl': ctx.losses['cl'].value()[0],
             'loss_ref': ctx.losses['reg'].value()[0], 
             'loss': ctx.losses['tot'].value()[0],
             'top1': ctx.errors.value()[0]
             }
    # batch_stats = {'loss': loss.item(), 
    #          'top1': ctx.errors.value()[0], 
    #          'top5': ctx.errors.value()[1]}

    # ctx.metrics.batch = stats
    ctx.metrics['avg'] = avg_stats
    #ctx.images = input
    return avg_stats

@epoch_hook(ctx, mode='train')
def train(train_loader, model, criterion, optimizer, epoch, opt):
    data_time = TimeMeter(unit=1)
    ctx.losses['cl'] = AverageValueMeter()
    ctx.losses['reg'] = AverageValueMeter()
    ctx.losses['tot'] = AverageValueMeter()
    ctx.errors = ClassErrorMeter(topk=[1])
    # switch to train mode
    if isinstance(model, list):
        for m in model:
            m.train()
    else:
        model.train()

    # end = time.time()
    for i, (win_past, inp_pred, labels, min_values) in enumerate(zip(*train_loaders)):
        # tmp var (for convenience)
        ctx.i = i
        win_past = win_past.cuda(opt['g'], non_blocking=True)
        inp_pred = inp_pred.cuda(opt['g'], non_blocking=True)
        labels = labels.cuda(opt['g'], non_blocking=True)
        min_values = min_values.cuda(opt['g'], non_blocking=True)
        inputs = (win_past, inp_pred)
        targets = (labels, min_values)
        stats = runner(inputs, targets, model, criterion, optimizer)

        loss = stats['loss']
        top1 = stats['top1']

        if i % opt['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {time:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Err@1 {top1:.3f}'.format(
                   epoch, i, len(train_loader),
                   time=data_time.value(), loss=loss, 
                   top1=top1))

    return stats
 
@epoch_hook(ctx, mode='val')
def validate(val_loader, model, criterion, opt):
    data_time = TimeMeter(unit=1)
    losses = AverageValueMeter()
    errors = ClassErrorMeter(topk=[1])
    # switch to evaluate mode
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (win_past, inp_pred, labels, min_values) in enumerate(zip(*val_loaders)):
            win_past = win_past.cuda(opt['g'], non_blocking=True)
            inp_pred = inp_pred.cuda(opt['g'], non_blocking=True)
            labels = labels.cuda(opt['g'], non_blocking=True)
            min_values = min_values.cuda(opt['g'], non_blocking=True)

            # compute output
            yh_cl, yh_reg = basik_tasks_model(win_past, win_pred)
            
            loss = criterion[0](yh_cl, labels)
            ctx.losses['cl'].add(loss.item())
            loss += opt['lambda'] * criterion[1](output, min_values)
            # measure accuracy and record loss
            errors.add(yh_cl.data, targets[0].data)
            losses['reg'].add(loss.item() - ctx.losses['cl'].value())
            losses['tot'].add(loss.item())
 
            errors.add(yh_cl, labels)
            losses.add(loss.item())
          
            loss = losses['tot'].value()[0]
            top1 = errors.value()[0]

            # if i % opt['print_freq'] == 0:
            #     print('[{0}/{1}]\t'
            #           'Time {time:.3f}\t'
            #           'Loss {loss:.4f}\t'
            #           'Err@1 {top1:.3f}\t'
            #           'Err@5 {top5:.3f}'.format(
            #            i, 
            #            len(val_loader),
            #            time=data_time.value(), loss=loss, 
            #            top1=top1, top5=top5))

        print('Loss {loss:.4f}'
              ' * Err@1 {top1:.3f}\t'
              .format(loss=loss, top1=top1))
    stats = {'loss': loss, 'top1': top1}
    ctx.metrics = stats
    return stats


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not ctx.opt['save']:
        return
    opt = ctx.opt
    fn = os.path.join(opt['o'], opt['arch'], opt['filename']) + '.pth.tar'
    r = gitrev(opt)
    meta = dict(SHA=r[0], STATUS=r[1], DIFF=r[2])
    state.update({'meta': meta})
    th.save(state, fn)
    if is_best:
        filename = os.path.join(opt['o'], opt['arch'], 
                            opt['filename']) + '_best.pth.tar'
        shutil.copyfile(fn, filename)


# adjust learning rate and log 
def adjust_learning_rate(epoch):
    opt = ctx.opt
    optimizer = ctx.optimizer

    if opt['lrs'] == '':
        # default lr schedule
        lr = opt['lr'] * (0.1 ** (epoch // 30))
    else:
        lr = schedule(ctx, k='lr')
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('[%s]: '%'lr', lr)
    if opt.get('fl', None) and ctx.ex.logger:
        ctx.ex.logger.info('[%s] '%'lr' + json.dumps({'%s'%'lr': lr}))


@train_hook(ctx)
def main_worker(opt):
    global best_top1

     # create model
    # if opt['pretrained']:
    #     print("=> using pre-trained model '{}'".format(opt['arch']))
    #     if 'allcnn' in opt['arch'] or 'wrn' in opt['arch']:
    #         model = models.__dict__[opt['arch']](opt)
    #         load_pretrained(model, opt['dataset'])
    #     else:
    #         model = models.__dict__[opt['arch']](opt, pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(opt['arch']))
    #     model = models.__dict__[opt['arch']](opt)
    
    ctx_net_opt = {'output_len': opt['win_pred_len'], 'num_channels': opt['ctx_ncls'], 'kernel_size': opt['ctx_ksz']}
    model = BasicTaskNet(ctx_net_opt, num_inputs=7, num_channels=opt['num_ncls'], kernel_size=opt['ksz'])
    if opt['ng'] == 0:
        torch.cuda.set_device(opt['g'])
        if isinstance(model, list):
            for m in model:
                m = m.cuda(opt['g'])
        else:
            model = model.cuda(opt['g'])
    else:
        model = torch.nn.DataParallel(model, 
                        device_ids=range(opt['g'], opt['g'] + opt['ng'],
                        output_device=opt['g'])).cuda()

    # define loss function (criterion) and optimizer
    criterion_cl = nn.CrossEntropyLoss().cuda(opt['g'])
    criterion_reg = nn.MSELoss().cuda(opt['g'])
    criterion = [criterion_cl, criterion_reg]
    optimizer = torch.optim.SGD(model.parameters(), opt['lr'],
                                momentum=opt['momentum'],
                                weight_decay=opt['wd'])

    ctx.optimizer = optimizer
    ctx.model = model

    # optionally resume from a checkpoint
    if opt['resume']:
        if os.path.isfile(opt['resume']):
            print("=> loading checkpoint '{}'".format(opt['resume']))
            checkpoint = torch.load(opt['resume'])
            opt['start_epoch'] = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt['resume']))

    cudnn.benchmark = True

    # Data loading code
    train_loaders, val_loaders = load_data(ctx)
    train_loaders, val_loaders = train_loaders['basic'], val_loaders['basic']
    if opt['evaluate']:
        validate(val_loaders, model, criterion, opt)
        return

    for epoch in range(opt['start_epoch'], opt['epochs']):
        ctx.epoch = epoch
        adjust_learning_rate(epoch)

        # train for one epoch
        train(train_loaders, model, criterion, optimizer, epoch, opt)

        # evaluate on validation set
        metrics = validate(val_loaders, model, criterion, opt)

        # remember best top@1 and save checkpoint
        top1 = metrics['top1']
        is_best = top1 < best_top1
        best_top1 = min(top1, best_top1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt['arch'],
            'state_dict': model.state_dict(),
            'best_top1': best_top1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


@ctx.ex.automain
def main():
    init()

    if ctx.opt['seed'] is not None:
        random.seed(ctx.opt['seed'])
        torch.manual_seed(ctx.opt['seed'])
        #cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')
    main_worker(ctx.opt)
