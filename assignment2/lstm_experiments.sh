python run_exp_lstm.py --exp_id lstm_l1_b16_adam      --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer adam
python run_exp_lstm.py --exp_id lstm_l1_b16_adamw     --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer adamw
python run_exp_lstm.py --exp_id lstm_l1_b16_sgd       --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer sgd
python run_exp_lstm.py --exp_id lstm_l1_b16_momentum  --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer momentum

python run_exp_lstm.py --exp_id lstm_l2_b16_adamw     --model lstm --layers 2 --batch_size 16 --log --epochs 10 --optimizer adamw
python run_exp_lstm.py --exp_id lstm_l4_b16_adamw     --model lstm --layers 4 --batch_size 16 --log --epochs 10 --optimizer adamw
