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


  def _build_model(self):
    hidden_size = self.hidden_size
    num_layers = self.num_layers
    keep_prob = self.keep_prob

    inputs = cntk.sequence.input_variable((69), name='inputs')
    targets = cntk.input_variable((69), name='targets')

    def lstm_cell():
      if self.config.bid == 1:
        _cell_creator = cntk.layers.Recurrence(cntk.layers.LSTM(hidden_size, use_peepholes=self.params.use_peephole), name='basic_lstm')
      else:
        _cell_creator = cntk.layers.Sequential([
          (cntk.layers.Recurrence(cntk.layers.LSTM(hidden_size, use_peepholes=self.params.use_peephole)),
          cntk.layers.Recurrence(cntk.layers.LSTM(hidden_size, use_peepholes=self.params.use_peephole), go_backwards=True)),
          cntk.ops.splice], name='bidirect_lstm')
      if self.params.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
      
    def gru_cell():
      if self.config.bid == 1:
        _cell_creator = cntk.layers.Recurrence(cntk.layers.GRU(hidden_size) , name='gru')
      else:
        _cell_creator = cntk.layers.Sequential([
          (cntk.layers.Recurrence(cntk.layers.GRU(hidden_size)), 
          cntk.layers.Recurrence(cntk.layers.GRU(hidden_size), go_backwards=True)), 
          cntk.ops.splice], name='bidirect_gru')
      if self.params.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
    
    def cifg_cell():
      if self.config.bid == 1:
        _cell_creator = cntk.layers.Recurrence(CIFG_LSTM(hidden_size, use_peepholes=self.params.use_peephole), name='cifg_lstm')
      else:
        _cell_creator = cntk.layers.Sequential([
          (cntk.layers.Recurrence(CIFG_LSTM(hidden_size, use_peepholes=self.params.use_peephole)),
          cntk.layers.Recurrence(CIFG_LSTM(hidden_size, use_peepholes=self.params.use_peephole), go_backwards=True)),
          cntk.ops.splice], name='bidirect_cifg')
      if self.params.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
        
    if self.config.cell == 'gru':
      _cell_creator = gru_cell
    elif self.config.cell == 'lstm':
      _cell_creator = lstm_cell
    elif self.config.cell == 'cifg_lstm':
      _cell_creator = cifg_cell
    else:
      raise ValueError("Unsupported cell type, choose from {'lstm', 'gru', 'cifg_lstm'}.")

    if self.params.use_residual and self.config.bid==1:
      print("  ** using residual **  ")
      _output = inputs
      for _ in range(num_layers):
        _output = self.params.resWeight * _cell_creator()(_output) + _output
        # _output = _cell_creator()(_output) + _output
    else:
      cell = cntk.layers.For(range(num_layers), lambda:_cell_creator())
      _output = cell(inputs)
    
    self.output = _output
    self.loss = cntk.squared_error(targets, output)
    # criterion = cntk.combine([loss, errs])
    # self.criterion = criterion
    cost = cntk.reduce_mean(loss, axis=cntk.Axis.all_axes()) 
    cost_rmse = cntk.reduce_l2(loss, axis=cntk.Axis.all_axes())   
    self.cost = cost


  @property
  def batch_size(self):
    return self.config.batch_size

  
  @property
  def hidden_size(self):
    return self.config.hidden_size

  
  @property
  def keep_prob(self):
    return self.config.keep_prob


  @property
  def num_layers(self):
    return self.config.num_layers
    