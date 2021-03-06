import numpy as np
import json, logging, os, subprocess
import time

def gitrev(opt):
    cmds = [['git', 'rev-parse', 'HEAD'],
            ['git', 'status'],
            ['git', 'diff']]
    rs = []
    for c in cmds:
        subp = subprocess.Popen(c,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        r, _ = subp.communicate()
        rs.append(r)

    rs[0] = rs[0].strip()
    return rs

def create_logger(ctx, idx=0):
    opt = ctx.opt
    if not opt.get('fl', None):
        return

    if len(opt.get('resume', '')) > 0:
        print('Retraining, will stop logging')
        return

    if opt.get('filename', None) is None:
        build_filename(ctx)

    d = os.path.join(opt.get('o'), opt['arch']) +'/logs'
    if not os.path.isdir(d):
        os.makedirs(d)
    fn = os.path.join(d, opt['filename']+'.log')
    l = logging.getLogger('%s'%idx)
    l.propagate = False

    fh = logging.FileHandler(fn)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    l.setLevel(logging.INFO)
    l.addHandler(fh)

    r = gitrev(opt)
    l.info('SHA %s'%r[0])
    l.info('STATUS %s'%r[1])
    l.info('DIFF %s'%r[2])

    l.info('')
    l.info('[OPT] ' + json.dumps(opt))
    l.info('')
    ctx.ex.logger = l
    return l


#def schedule(opt, e, logger=None, k=None):
def schedule(ctx, k=None):
    logger = ctx.ex.logger
    e = ctx.epoch
    opt = ctx.opt
    ks = k + 's'
    if opt[ks] == '':
        opt[ks] = json.dumps([[opt['B'], opt[k]]])

    rs = json.loads(opt[ks])

    idx = len(rs)-1
    for i in range(1,len(rs)):
        if e < rs[i][0]:
            idx = i-1
            break
    if e >= rs[len(rs)-1][0]:
        idx = i

    r = rs[idx][1]
    return r


def init_opt(ctx):
    cfg = ctx.ex.current_run.config
    opt = dict()
    for k,v in cfg.items():
        opt[k] = v
    if len(ctx.ex.named_configs) > 0:
        for k,v in cfg.items():
            opt[k] = v
    return opt

def build_filename(ctx):
    opt = ctx.opt
    whitelist = opt['whitelist']
    marker = opt['marker']
    dconf = dict()
    cfg_mdf = ctx.ex.current_run.config_modifications.modified
    for k in cfg_mdf:
        dconf[k] = opt[k]

    base_whilelist = ['dataset', 'arch']
    blacklist = ['fl', 'tfl', 'dbl', 'o']
    
    from ast import literal_eval
    whitelist = literal_eval(whitelist)
    whitelist += base_whilelist 
    
    o = json.loads(json.dumps(opt))
    # dconf overwrites if same key
    o = {**o, **dconf}
    oc = o.copy()
    for k in o.keys():
        if k not in whitelist or k in blacklist:
            oc.pop(k, None)
    o = oc
    t = ''
    if not marker == '':
        t = marker + '_'
    t = t + time.strftime('(%b_%d_%H_%M_%S)') + '_opt_'
    opt['time'] = t
    opt['filename'] = t + json.dumps(o, sort_keys=True,
                separators=(',', ':'))
