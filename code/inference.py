import click

data_Y = "Y_output_69_k2.mat"
X_input = sio.loadmat(os.path.join(file_path, data_X))['X_input']
Y_output = sio.loadmat(os.path.join(file_path, data_Y))['Y_output']

class Predictor:
    def __init__(self, params, configs):
        self.params = params
        self.config = config
        self._build_graph()


@click.command()

# for params
@click.option('--load_model_dir', default=None, type=str)
@click.option('--load_model_name', default=None, type=str)
@click.option('--data_path', default=None, type=str, help="Where the training/test data is stored.")
@click.option('--data_in', default=None, type=str, help='name of the test input file')
@click.option('--data_out', default=None, type=str, help='name of the test output file')

# for config
@click.option('--loss_fct', default="sampledsoftmax", type=str, help="The loss function to use. Choose among softmax, sampledsoftmax")
@click.option('--use_dropout', default=True, type=bool)
@click.option('--use_highway', default=False, type=bool)
@click.option('--use_residual', default=False, type=bool)
@click.option('--use_peephole', default=False, type=bool)
@click.option('--res_weight', default=1, type=float, help="output=resWeight*res+input")
@click.option('--learning_rate', default=0.01, type=float)
@click.option('--lr_decay', default=0.9, type=float)
@click.option('--keep_prob', default=0.7, type=float, help="Dropout keep probability")
@click.option('--grad_clip', default=2.3, type=float)
@click.option('--bid', default=1, type=int, help="1 means single direction RNN, 2 means bidirection RNN")
@click.option('--batch_size', default=1, type=int)
@click.option('--num_layers', default=2, type=int)
@click.option('--hidden_size', default=512, type=int)
@click.option('--output_size', default=69, type=int)
@click.option('--cell', default="lstm", type=str, help="Choose among lstm, gru, cifg_lstm")

def main(   
            load_model_dir, load_model_name, data_path, log_dir, data_in, data_out,
            loss_fct, use_dropout, use_highway, use_residual, use_peephole, res_weight, 
            learning_rate, lr_decay, keep_prob, grad_clip, bid, 
            batch_size, num_layers, hidden_size, output_size, cell,
        ):
  params = {}
  params["load_model_dir"] = load_model_dir
  params["load_model_name"] = load_model_name
  params["data_path"] = data_path
  params["data_in"] = data_in
  params["data_out"] = data_out

  configs = {}
  configs["loss_fct"] = loss_fct
  configs["use_dropout"] = use_dropout
  configs["use_highway"] = use_highway
  configs["use_residual"] = use_residual
  configs["use_peephole"] = use_peephole
  configs["resWeight"] = res_weight
  configs["learning_rate"] = learning_rate
  configs["lr_decay"] = lr_decay
  configs["keep_prob"] = keep_prob
  configs["grad_clip"] = grad_clip
  configs["bid"] = bid
  configs["batch_size"] = batch_size
  configs["num_layers"] = num_layers
  configs["hidden_size"] = hidden_size
  configs["output_size"] = output_size
  configs["cell"] = cell
  _config = Config(flag="configs", params=configs)
  _params = Config(flag="params", params=params)
  ll = Predictor(_config, _params)
  ll()

if __name__ == "__main__":
  main()
