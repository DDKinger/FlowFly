import cntk
from cifg_lstm import CIFG_LSTM

class Model(object):
  def __init__(self, config):
    self.config = config
    self._build_model()


  def _build_model(self):
    hidden_size = self.hidden_size
    output_size = self.output_size
    num_layers = self.num_layers
    keep_prob = self.keep_prob

    inputs = cntk.sequence.input_variable((output_size), name='inputs')
    target = cntk.input_variable((output_size), name='target')
    self.inputs = inputs
    self.target = target

    if self.config.use_embedding:
      inputs = cntk.layers.Dense(self.config.embedding_size)(inputs)

    def lstm_cell():
      _cell_creator = cntk.layers.Recurrence(cntk.layers.LSTM(hidden_size, use_peepholes=self.config.use_peephole), name='basic_lstm')
      if self.config.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
      
    def gru_cell():
      _cell_creator = cntk.layers.Recurrence(cntk.layers.GRU(hidden_size) , name='gru')
      if self.config.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
    
    def cifg_cell():
      _cell_creator = cntk.layers.Recurrence(CIFG_LSTM(hidden_size, use_peepholes=self.config.use_peephole), name='cifg_lstm')
      if self.config.use_dropout:
        print("  ** using dropout for LSTM **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator

    def rnn_cell():
      _cell_creator = cntk.layers.Recurrence(cntk.layers.RNNStep(hidden_size), name='rnn')
      if self.config.use_dropout:
        print("  ** using dropout for RNN **  ")
        _cell_creator = cntk.layers.Dropout(keep_prob = keep_prob)(_cell_creator)
      return _cell_creator
        
    if self.config.cell == 'gru':
      _cell_creator = gru_cell
      # _cell_creator_last = cntk.layers.Recurrence(cntk.layers.GRU(output_size) , name='gru_last')
    elif self.config.cell == 'lstm':
      _cell_creator = lstm_cell
      # _cell_creator_last = cntk.layers.Recurrence(cntk.layers.LSTM(output_size) , name='lstm_last')
    elif self.config.cell == 'cifg_lstm':
      _cell_creator = cifg_cell
      # _cell_creator_last = cntk.layers.Recurrence(CIFG_LSTM(output_size) , name='cifg_last')
    elif self.config.cell == 'rnn':
      _cell_creator = rnn_cell
    else:
      raise ValueError("Unsupported cell type, choose from {'lstm', 'gru', 'cifg_lstm', 'rnn'}.")

    if self.config.use_residual:
      if num_layers < 2:
        raise ValueError("if using residual connection, num_layers should be greater than 1")
      print("  ** using residual **  ")
      _output = _cell_creator()(inputs)
      for _ in range(num_layers-1):
        _output = self.config.resWeight * _cell_creator()(_output) + _output
        # _output = _cell_creator()(_output) + _output
    else:
      cell = cntk.layers.For(range(num_layers), lambda:_cell_creator())
      _output = cell(inputs)
    
    # _output = _cell_creator_last(_output)
    output = cntk.sequence.last(_output)
    if self.config.cell == 'cifg_lstm':
      output = cntk.layers.Dense(output_size, name = 'y_out')(output)
    else:
      output = cntk.layers.Dense(output_size, activation=cntk.sigmoid, name = 'y_out')(output)
    self.output = output
    loss = cntk.squared_error(output, target)
    self.loss = loss
    cost_mape = cntk.reduce_mean(cntk.abs(output-target)/target, axis=cntk.Axis.all_axes(), name='mape') 
    cost_mae = cntk.reduce_mean(cntk.abs(output-target), axis=cntk.Axis.all_axes(), name='mae')
    cost_rmse = cntk.sqrt(cntk.reduce_mean(cntk.square(output-target), axis=cntk.Axis.all_axes()), name='rmse')
    self.cost = cntk.combine([cost_mape, cost_mae, cost_rmse])
    # self.criterion = cntk.combine([loss, cost_mape])


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
    