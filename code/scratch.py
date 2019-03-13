from sacred import Experiment
 
ex = Experiment('b')

@ex.config
def cfg():
    a = 4

@ex.named_config
def cfg1():
    b = 1

@ex.named_config
def cfg2():
    c = 1

@ex.automain
@ex.capture
def main(_config,c):
    print(ex)
