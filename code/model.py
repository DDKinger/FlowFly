import cntk

class Model(object):
    def __init__(self, config, is_training = True, loss_fct="softmax"):
        self.config = config
        self.loss_fct = loss_fct
        self.is_training = is_training

    if self.config.bid==1:
      self.output_hidden_size=config.hidden_size 
    else:
      self.output_hidden_size=config.hidden_size *2
    self._build_model()