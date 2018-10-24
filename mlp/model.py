import cntk

class Model(object):
  def __init__(self, config):
    self.config = config
    self._build_model()


  def _build_model(self):
    hidden_size = self.hidden_size
    output_size = self.output_size
    num_layers = self.num_layers
    keep_prob = self.keep_prob
    ts =self.config.time_step

    inputs = cntk.input_variable((ts*output_size), name='inputs')
    target = cntk.input_variable((output_size), name='target')
    self.inputs = inputs
    self.target = target

    def cell_creator():
      cell = cntk.layers.Dense(hidden_size, activation=cntk.sigmoid)
      if self.config.use_dropout:
        cell = cntk.layers.Dropout(keep_prob=keep_prob)(cell)
      return cell

    cell = cntk.layers.For(range(num_layers), lambda:cell_creator())
    output = cell(inputs)
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
    