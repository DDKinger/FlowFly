import cntk

class Model(object):
  def __init__(self, config):
    self.config = config


  def _build_model(self):
    hidden_size = self.hidden_size
    output_size = self.output_size
    num_layers = self.num_layers
    keep_prob = self.keep_prob

    inputs = cntk.sequence.input_variable((output_size), name='inputs')
    target = cntk.input_variable((output_size), name='target')

    def lstm_cell():
      _cell_creator = cntk.layers.Recurrence(cntk.layers.LSTM(hidden_size, use_peepholes=self.params.use_peephole), name='basic_lstm')
      if self.params.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
      
    def gru_cell():
      _cell_creator = cntk.layers.Recurrence(cntk.layers.GRU(hidden_size) , name='gru')
      if self.params.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
    
    def cifg_cell():
      _cell_creator = cntk.layers.Recurrence(CIFG_LSTM(hidden_size, use_peepholes=self.params.use_peephole), name='cifg_lstm')
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

    if self.params.use_residual:
      print("  ** using residual **  ")
      _output = inputs
      for _ in range(num_layers):
        _output = self.params.resWeight * _cell_creator()(_output) + _output
        # _output = _cell_creator()(_output) + _output
    else:
      cell = cntk.layers.For(range(num_layers), lambda:_cell_creator())
      _output = cell(inputs)
    
    _output = cntk.sequence.last(_output)
    output = cntk.layers.Dense(output_size)(_output)
    self.output = output
    self.loss = cntk.squared_error(output, target)
    cost_mape = cntk.reduce_mean(cntk.abs(output-target)/target, axis=cntk.Axis.all_axes(), name='mape') 
    cost_mae = cntk.reduce_mean(cntk.abs(output-target), axis=cntk.Axis.all_axes(), name='mae')
    cost_rmse = cntk.reduce_l2((output-target), axis=cntk.Axis.all_axes(), name='rmse')  
    self.cost = cntk.combine([cost_mape, cost_mae, cost_rmse])
    self.criterion = cntk.combine([loss, cost_mape])


  @property
  def output_size(self):
    return self.config.output_size
  
  @property
  def hidden_size(self):
    return self.config.hidden_size

  
  @property
  def keep_prob(self):
    return self.config.keep_prob


  @property
  def num_layers(self):
    return self.config.num_layers
    