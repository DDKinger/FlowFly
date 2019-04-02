import numpy as np
import scipy.io as sio
import click

@click.command()
@click.option('--ts', default=None, type=int, required=True)
@click.option('--interval', default=0, type=int)
def main(ts, interval):
    data_dir = "./TrafficFlow_69_12week_6day"
    x = sio.loadmat(data_dir)['traffic_flow'] 
    print("traffic_flow shape:", x.shape)
    # n_station = 69
    if interval > 576:
        n_weekday = 576    # 12*24*2
    elif interval > 288:
        n_weekday = 864    # 12*24*3
    elif interval > 72:
        n_weekday = 1152    # 12*24*4
    else:
        n_weekday = 1440    # 12*24*5
    n_week = 1728   # 12*24*6
    row = n_weekday*12
    _x = []
    _y = []

    for i in range(0, n_week*11+1, n_week):       
        for j in range(i, i+n_weekday):           
            _x.append(x[j:j+ts])
            _y.append(x[j+ts+interval])
    _x = np.asarray(_x, dtype=np.float)
    _y = np.asarray(_y, dtype=np.float)
    print("_x.shape:", _x.shape)
    print("_y.shape:", _y.shape)
    X_input = _x.reshape(row, -1, order='F')
    Y_output = _y.reshape(row, -1, order='F')
    print("X_input.shape:", X_input.shape)
    print("Y_output.shape:", Y_output.shape)
    print("n_weekday:", n_weekday)
    sio.savemat('../X_input_69_k'+ str(ts)+'_h'+str(interval), {'X_input': X_input})
    sio.savemat('../Y_output_69_k'+ str(ts)+'_h'+str(interval), {'Y_output': Y_output})


if __name__ == "__main__":
    main()