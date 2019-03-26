cd DD/ML_Project/Traffic_flow/code/

python train.py --model_dir=../models --save_model_name=flow_prediction --data_dir=../data --log_dir=../logs --train_epoch=2000 --batch_size=144 --learning_rate=0.08 --time_step=8 --num_layers=2 --hidden_size=512 --output_size=69 --cell=lstm --use_residual=True --res_weight=0.1 --use_dropout=True --keep_prob=0.9 --use_peephole=False

python train.py --model_dir=..\..\models --save_model_name=flow_prediction --data_dir=..\..\data --log_dir=..\..\logs --use_dropout=False --train_epoch=4 --batch_size=144 --time_step=8 --num_layers=1 --hidden_size=128 --output_size=69 --cell=lstm

# mlp
python train.py --model_dir=./models --save_model_name=mlp --data_dir=../data --log_dir=./logs --use_dropout=False --train_epoch=2000 --batch_size=144 --time_step=2 --num_layers=2 --hidden_size=256 --output_size=69

# mlp_keras
python mlp_keras.py --data_dir=../data --train_epoch=2000 --batch_size=144 --time_step=6 --num_layers=2 --hidden_size=512 --output_size=69 --keep_prob=0.5 --optimizer=adam