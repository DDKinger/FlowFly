import numpy as np
import scipy.io as sio
import click

@click.command()
@click.option('--ts', default=None, type=int, required=True)
@click.option('--time_len', default=None, type=int, required=True)
def main(ts, time_len):
    data_dir = "D:/missdd/ML_Project/Traffic_Flow_Prediction_LSTM/data/TrafficFlow_69_12week_6day"
    # data_dir = "D:/dataset/Traffic/镇江/mat/16P/ZJ_TrafficFlow_10T_16P"
    x = sio.loadmat(data_dir)['traffic_flow'] 
    print("traffic_flow shape:", x.shape)
    # n_station = 69

    n_day = 60 // time_len * 24
    n_weekday = n_day * 5
    n_week = n_day * 6
    print(n_week, type(n_week))

    week = 12
    row = n_weekday*week
    _x = []
    _y = []

    for i in range(0, n_week*(week-1)+1, n_week):       
        for j in range(i, i+n_weekday):
            _x.append(x[j:j+ts])
            _y.append(x[j+ts])
    _x = np.asarray(_x, dtype=np.float)
    _y = np.asarray(_y, dtype=np.float)
    print("_x.shape:", _x.shape)
    print("_y.shape:", _y.shape)
    X_input = _x.transpose(0,2,1).reshape(-1,ts)
    Y_output = _y.reshape((-1, 1))
    print("X_input.shape:", X_input.shape)
    print("Y_output.shape:", Y_output.shape)
    sio.savemat('./X_input_69_k'+ str(ts), {'X_input': X_input})
    sio.savemat('./Y_output_69_k'+ str(ts), {'Y_output': Y_output})


if __name__ == "__main__":
    main()