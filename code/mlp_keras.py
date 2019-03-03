import numpy as np
#import pickle
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import click
from keras import backend as K
from data_read_keras import get_xy

def run(params):
    norm = 1188
    n_obj = params['output_size']
    ts = params['time_step']
    drop_rate = 1-params['keep_prob']
    data_dir = params['data_dir']
    dg = params['data_gaussian']
    df = params['data_shuffle']
    bs = params['batch_size']
    ts = params['time_step']
    x_train, y_train, vmax = get_xy(data_dir, True, dg, df, bs, ts)
    x_valid, y_valid, _ = get_xy(data_dir, False, dg, df, bs, ts)
    x_input = Input(shape=(ts*n_obj,))
    for i in range(params['num_layers']):
        if i == 0:
            x = Dense(params['hidden_size'], activation='sigmoid')(x_input)
        else:
            x = Dense(params['hidden_size'], activation='sigmoid')(x)
        if params['use_dropout']:
            x = Dropout(drop_rate)(x)
    y_output = Dense(n_obj, activation='sigmoid', name = 'y_out')(x)
    model = Model(x_input, y_output)
    model.summary()
    model.compile(loss='mean_squared_error',
              optimizer=params['optimizer'],
              metrics=['mean_absolute_percentage_error'])

    history = model.fit(x_train, y_train,
                        validation_data = (x_valid, y_valid),
                        batch_size=bs, nb_epoch=params['train_epoch'],
                        verbose=0, shuffle=False)
    score = model.evaluate(x_valid, y_valid, verbose=0)
    y_predict = model.predict(x_valid)
    MAE_test = np.mean(np.abs((y_valid - y_predict)))*norm
    RMSE_test = np.sqrt(np.mean((y_valid - y_predict)**2))*norm

    print('Test score:', score[0])
    print('MAPE_test:', score[1])
    print('MAE_test:', MAE_test)
    print('RMSE_test:', RMSE_test)

@click.command()

# for params
@click.option('--data_dir', default=None, type=str)
@click.option('--data_gaussian', default=False, type=bool)
@click.option('--data_shuffle', default=True, type=bool)
@click.option('--use_dropout', default=True, type=bool)
@click.option('--learning_rate', default=0.01, type=float)
@click.option('--lr_decay', default=0.9, type=float)
@click.option('--keep_prob', default=0.7, type=float, help="Dropout keep probability")
@click.option('--grad_clip', default=2.3, type=float)
@click.option('--train_epoch', default=8, type=int, help="training epoch")
@click.option('--batch_size', default=144, type=int)
@click.option('--time_step', default=8, type=int)
@click.option('--num_layers', default=2, type=int)
@click.option('--hidden_size', default=128, type=int)
@click.option('--output_size', default=69, type=int)
@click.option('--optimizer', default='adagrad', type=str)

def main( 
          data_dir, data_gaussian, data_shuffle, use_dropout, optimizer,
          learning_rate, lr_decay, keep_prob, grad_clip, train_epoch,
          batch_size, time_step, num_layers, hidden_size, output_size):
  params = {}
  params["data_dir"] = data_dir
  params["data_gaussian"] = data_gaussian
  params["data_shuffle"] = data_shuffle
  params["use_dropout"] = use_dropout
  params["learning_rate"] = learning_rate
  params["lr_decay"] = lr_decay
  params["keep_prob"] = keep_prob
  params["grad_clip"] = grad_clip
  params["train_epoch"] = train_epoch
  params["batch_size"] = batch_size
  params["time_step"] = time_step
  params["num_layers"] = num_layers
  params["hidden_size"] = hidden_size
  params["output_size"] = output_size
  params["optimizer"] = optimizer
  run(params)


if __name__ == "__main__":
  main()