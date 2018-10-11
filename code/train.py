import os
import click
import cntk
from data_read import get_xy
from model import Model
from config import Config

class Train:
    def __init__(self, params, configs):
        self.params = params
        self.config = config
        self._build_graph()


    def _load_data(self):
        data_dir = self.params.data_dir
        dg = self.params.data_gaussian
        df = self.params.data_shuffle
        bs = self.config.batch_size
        ts = self.config.time_step
        self.x_train, self.y_train, self.vmax = get_xy(data_dir, is_training=True, dg, df, bs, ts)
        self.x_valid, self.y_valid, _ = get_xy(data_dir, is_training=False, dg, df, bs, ts)


    def _build_graph(self):
        self.train_model = Model(self.config)


    def _run(self):
        m = self.train_model
        config = self.config
        params = self.params
        vmax = self.vmax
        output = m.output
        _inputs = m.inputs
        _target = m.target
        criterion = m.criterion
        cost = m.cost
        lr = config.learning_rate
        last_epoch = 0
        last_step = 0
        if not os.path.exists(params.model_dir):
            os.makedirs(params.model_dir)
        if not os.path.exists(params.log_dir):
            os.makedirs(params.log_dir) 

        def model_path(epoch):
            model_path = os.path.join(params.model_dir, params.save_model_name) + ".cmf." + str(epoch)
            return model_path

        if params.continue_training:
            load_model = os.path.join(params.load_model_dir, params.load_model_name)
            if os.path.exists(load_model):   
                last_epoch = int(params.load_model_name.split(".")[-1])
                logits.restore(load_model)
                print("[INFO] Restore from %s at Epoch %d" % (params.load_model_name, last_epoch))
            else:
                raise ValueError("'load model path' can't be found")

        print()
        print("_____________Starting training______________")
        cntk.logging.log_number_of_parameters(logits)
        print(output.parameters)
        print()
        # learner = cntk.learners.fsadagrad(
        #     output.parameters,
        #     lr = cntk.learners.learning_parameter_schedule_per_sample([lr]*2+[lr/2]*3+[lr/4], epoch_size=self.config.train_epoch),
        #     momentum = cntk.learners.momentum_schedule_per_sample(0.9),
        #     gradient_clipping_threshold_per_sample = config.grad_clip,
        #     gradient_clipping_with_truncation = True
        # )
        learner = cntk.learners.adagrad(
            output.parameters,
            lr = cntk.learners.learning_parameter_schedule_per_sample([lr]*2+[lr/2]*3+[lr/4], epoch_size=self.config.train_epoch),
            gradient_clipping_threshold_per_sample = config.grad_clip,
            gradient_clipping_with_truncation = True
        )
        progress_log_file = os.path.join(params.model_dir, params.save_model_name) + ".txt"
        progress_writer = cntk.logging.ProgressPrinter(freq=params.print_freq, tag='Training', log_to_file=progress_log_file)
        tensorboard_writer = cntk.logging.TensorBoardProgressWriter(log_dir=params.log_dir, model=output)
        trainer = cntk.Trainer(None, criterion, learner, progress_writers=tensorboard_writer)
        start_time = time.time()
        localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        print(localtime)

        if last_epoch < config.train_epoch:
            for epoch in range(last_epoch, config.train_epoch): 
                print("---------------------------------") 
                print("epoch ", epoch+1, " start")   
                for step, (x, y) in enumerate(zip(self.x_train, self.y_train)):
                    trainer.train_minibatch({_inputs: x, _targets: y})
                    progress_writer.update_with_trainer(trainer, with_metric=True)
                progress_writer.epoch_summary(with_metric=True)
                output.save(model_path(epoch+1)
                print("Saving model to '%s'" % model_path(epoch+1))
                
                costs = {'mape':[], 'mae':[], 'rmse:',[]}
                _costs = cost.eval({_inputs:x_v, _targets:y_v})
                for key in _costs:
                    if key_name = 'mape'
                        costs[key.name].append(_costs[key])   
                    else:
                        costs[key.name].append(_costs[key]*vmax)            
                print("valid_mape:", costs['mape'][-1])
                print("valid_mae:", costs['mae'][-1])
                print("valid_rmse:", costs['rmse'][-1]))
                
            print()
            print("train time:", time.time()-start_time)
            localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            print(localtime)

            y_predict = output.eval({_inputs:x_v, _targets:y_v})
            fileout = os.path.join(params.model_dir, 'results.txt')
            sio.savemat(fileout, {'y_predict':y_predict, 'costs':costs})

        else:
            print("the loaded model has been trained equal to or more than max_max_epoch")



@click.command()

# for params
@click.option('--model_dir', default=None, type=str)
@click.option('--save_model_name', default=None, type=str)
@click.option('--load_model_dir', default=None, type=str)
@click.option('--load_model_name', default=None, type=str)
@click.option('--data_dir', default=None, type=str, help="Where the training/test data is stored.")
@click.option('--log_dir', default=None, type=str, help="Where is the TensorBoard log data")
@click.option('--continue_training', default=False, type=bool, help="Continue training where it stopped")
@click.option('--data_gaussian', default=False, type=bool)
@click.option('--data_shuffle', default=True, type=bool)

# for config
@click.option('--use_dropout', default=True, type=bool)
@click.option('--use_highway', default=False, type=bool)
@click.option('--use_residual', default=False, type=bool)
@click.option('--use_peephole', default=False, type=bool)
@click.option('--res_weight', default=1, type=float, help="output=resWeight*res+input")
@click.option('--learning_rate', default=0.01, type=float)
@click.option('--lr_decay', default=0.9, type=float)
@click.option('--keep_prob', default=0.7, type=float, help="Dropout keep probability")
@click.option('--grad_clip', default=2.3, type=float)
@click.option('--train_epoch', default=8, type=int, help="training epoch")
@click.option('--batch_size', default=128, type=int)
@click.option('--time_step', default=8, type=int)
@click.option('--num_layers', default=2, type=int)
@click.option('--hidden_size', default=512, type=int)
@click.option('--output_size', default=69, type=int)
@click.option('--cell', default="lstm", type=str, help="Choose among lstm, gru, cifg_lstm")

def main( 
          model_dir, save_model_name, load_model_dir, load_model_name,
          data_dir, log_dir, continue_training, data_gaussian, data_shuffle,
          use_dropout, use_highway, use_residual, use_peephole, res_weight,
          learning_rate, lr_decay, keep_prob, grad_clip, train_epoch,
          batch_size, time_step, num_layers, hidden_size, output_size, cell):
  params = {}
  params["model_dir"] = model_dir
  params["save_model_name"] = save_model_name
  params["load_model_dir"] = load_model_dir
  params["load_model_name"] = load_model_name
  params["data_dir"] = data_dir
  params["log_dir"] = log_dir
  params["continue_training"] = continue_training
  params["data_gaussian"] = data_gaussian
  params["data_shuffle"] = data_shuffle
  configs = {}
  configs["use_dropout"] = use_dropout
  configs["use_highway"] = use_highway
  configs["use_residual"] = use_residual
  configs["use_peephole"] = use_peephole
  configs["resWeight"] = res_weight
  configs["learning_rate"] = learning_rate
  configs["lr_decay"] = lr_decay
  configs["keep_prob"] = keep_prob
  configs["grad_clip"] = grad_clip
  configs["train_epoch"] = max_max_epoch
  configs["batch_size"] = batch_size
  configs["time_step"] = time_step
  configs["num_layers"] = num_layers
  configs["hidden_size"] = hidden_size
  configs["output_size"] = output_size
  configs["cell"] = cell
  _config = Config(flag="configs", params=configs)
  _params = Config(flag="params", params=params)
  train = Train(_config, _params)
  train()

if __name__ == "__main__":
  main()