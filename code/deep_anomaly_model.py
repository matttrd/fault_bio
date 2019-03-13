from tcn import TemporalConvNet
import torch.nn as nn

class ContextNet(nn.Module):
    def __init__(self, num_inputs, output_len, num_channels, kernel_size=2, dropout=0.2):
        super(ContextNet, self).__init__()
        self.m = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        #for now just take last N_ctx samples
        #self.downsample = nn.Conv1d(num_channels[-1], num_channels[-1], 1) if n_inputs != n_outputs else None
    
    def forward(self, input):
        return self.m(input)[...,:-output_len]

class BasicTaskNet(nn.Module):
    # ctx_net_opt is a dict with num_inputs, output_len, num_channels, kernel_size, dropout
    #def __init__(self, ctx_net_opt, num_inputs, output_len, num_channels, kernel_size=2, dropout=0.2):
    def __init__(self, ctx_net_opt, num_inputs, num_channels, kernel_size=2, dropout=0.2):    
        super(BasicTaskNet, self).__init__()
        #self.ctx_net = ContextNet(**ctx_net_opt)
        self.ctx_net = ContextNet(num_inputs=5, **ctx_net_opt)
        self.common_net = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.regression = nn.Linear(num_channels[-1], 1)
        self.classification = nn.Linear(num_channels[-1], 2)

    def forward(self, input_ctx, input):
        y1 = self.ctx_net(input_ctx)
        yc = th.concat([y1, input], dim=1)
        y2 = self.common_net(yc)
        y_reg = self.regression(y2[..., -1])
        y_cl = self.classification(y2[..., -1])
        return y_cl, y_reg
