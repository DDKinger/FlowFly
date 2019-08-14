cd DD/ML_Project/Traffic_flow/code/

python train.py --model_dir=./models --save_model_name=ZJflow_prediction_pems --data_dir=./data --log_dir=./logs --batch_size=96 --learning_rate=0.08 --time_step=6 --num_layers=2 --hidden_size=512 --cell=lstm --use_residual=True --res_weight=0.1 --use_dropout=True --keep_prob=0.9 --data_shuffle=True --train_epoch=120000 --output_size=1 --data_name=ZJ_TrafficFlow_20T_16P --time_len=20 --continue_training=True --load_model_dir=./data --load_model_name=ZJflow_prediction_pems.cmf.59800

python train.py --model_dir=./models --save_model_name=ZJflow_prediction --data_dir=./data --log_dir=./logs --batch_size=128 --learning_rate=0.08 --time_step=6 --num_layers=2 --hidden_size=512 --cell=lstm --use_residual=True --res_weight=0.1 --use_dropout=True --keep_prob=0.9 --train_epoch=4000 --output_size=19 --data_name=ZJ_TrafficFlow_5T_19P --time_len=5

--use_peephole=False --continue_training=True --load_model_dir=./models --load_model_name=ZJflow_prediction.cmf.6000

python train.py --model_dir=../models --save_model_name=flow_prediction --data_dir=../data/NoneZero --log_dir=../logs --train_epoch=2000 --batch_size=144 --learning_rate=0.08 --time_step=6 --num_layers=2 --hidden_size=512 --output_size=69 --cell=lstm --use_residual=True --res_weight=0.1 --use_dropout=True --keep_prob=0.9 --use_peephole=False

python train.py --model_dir=..\..\models --save_model_name=flow_prediction --data_dir=..\..\data --log_dir=..\..\logs --use_dropout=False --train_epoch=4 --batch_size=144 --time_step=6 --num_layers=1 --hidden_size=128 --output_size=69 --cell=lstm

# mlp
python train.py --model_dir=./models --save_model_name=mlp --data_dir=../data/NoneZero --log_dir=./logs --use_dropout=True --keep_prob=0.9 --train_epoch=2000 --batch_size=144 --time_step=6 --num_layers=2 --hidden_size=512 --output_size=69

# mlp_keras
python mlp_keras.py --data_dir=../data --train_epoch=2000 --batch_size=144 --time_step=6 --num_layers=2 --hidden_size=512 --output_size=69 --keep_prob=0.5 --optimizer=adam