import os
import click
import cntk

class Train:
    def __init__(self, params, configs):
        self.params = params
        self.config = config
        self._build_graph()

    def _load_data(self):
        dataset = os.path.join(self.params.data_path, self.params.data_in)
        if not os.path.exists(dataset+'.mat'):
            raise ValueError("data path or filename is not valid")
        if not os.path.exists(dataset+'_norm.mat'):
            data_process.generate_data(dataset, self.params.data_shuffle, self.params.data_Gaussian, is_training=True)

@click.command()

# for params
@click.option('--model_dir', default=None, type=str)
@click.option('--save_model_name', default=None, type=str)
@click.option('--load_model_dir', default=None, type=str)
@click.option('--load_model_name', default=None, type=str)
@click.option('--data_path', default=None, type=str, help="Where the training/test data is stored.")
@click.option('--data_in', default=None, type=str, help='name of the input file')
@click.option('--data_out', default=None, type=str, help='name of the output file')
@click.option('--log_dir', default=None, type=str, help="Where is the TensorBoard log data")
@click.option('--print_freq', default=500, type=int, help="How many steps everytime the loss is printed")
@click.option('--log_freq', default=100, type=int, help="How many steps everytime the log is printed to TensorBoard")
@click.option('--save_rate', default=10000, type=int, help="How many steps everytime the model is saved")
@click.option('--continue_training', default=False, type=bool, help="Continue training where it stopped")
@click.option('--data_gaussian', default=False, type=bool)
@click.option('--data_shuffle', default=True, type=bool)
@click.option('--k_interval', default=2, type=int)


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
@click.option('--train_epoch', default=8, type=int, help="training epoch")
@click.option('--batch_size', default=128, type=int)
@click.option('--num_layers', default=2, type=int)
@click.option('--hidden_size', default=512, type=int)
@click.option('--output_size', default=69, type=int)
@click.option('--cell', default="lstm", type=str, help="Choose among lstm, gru, cifg_lstm")

def main( 
          model_dir, save_model_name, load_model_dir, load_model_name,
          data_path, data_in, data_out, log_dir, loss_fct, continue_training,
          use_dropout, use_highway, use_residual, use_peephole, res_weight,
          print_freq, log_freq, save_rate, backward_train,
          learning_rate, lr_decay, keep_prob, grad_clip, bid, max_max_epoch,
          batch_size, num_layers, embed_dim, hidden_size, fast_test, cell):
  params = {}
  params["model_dir"] = model_dir
  params["save_model_name"] = save_model_name
  params["load_model_dir"] = load_model_dir
  params["load_model_name"] = load_model_name
  params["data_path"] = data_path
  params["log_dir"] = log_dir
  params["loss_fct"] = loss_fct
  params["word_break_level"] = word_break_level
  params["bpe_separator"] = bpe_separator
  params["continue_training"] = continue_training
  params["use_fasttext_embedding"] = use_fasttext_embedding
  params["use_dropout"] = use_dropout
  params["use_highway"] = use_highway
  params["use_residual"] = use_residual
  params["use_peephole"] = use_peephole
  params["resWeight"] = res_weight
  params["vocab_size"] = vocab_size
  params["bpe_symbols"] = bpe_symbols
  params["bpe_min_frequency"] = bpe_min_frequency
  params["sampled_softmax_size"] = sampled_softmax_size
  params["nce_num_samples"] = nce_num_samples
  params["number_limit"] = number_limit
  params["print_freq"] = print_freq
  params["log_freq"] = log_freq
  params["save_rate"] = save_rate
  params["backward_train"] = backward_train
  configs = {}
  configs["learning_rate"] = learning_rate
  configs["lr_decay"] = lr_decay
  configs["keep_prob"] = keep_prob
  configs["grad_clip"] = grad_clip
  configs["bid"] = bid
  configs["max_max_epoch"] = max_max_epoch
  configs["batch_size"] = batch_size
  configs["num_layers"] = num_layers
  configs["embed_dim"] = embed_dim
  configs["hidden_size"] = hidden_size
  configs["fast_test"] = fast_test
  configs["cell"] = cell
  _config = Config(flag="configs", params=configs)
  _params = Config(flag="params", params=params)
  train = Train(_config, _params)
  train()

if __name__ == "__main__":
  main()