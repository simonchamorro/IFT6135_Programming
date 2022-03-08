
python run_exp_vit.py --log --exp_id vit_l2_b128_adam      --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer adam
python run_exp_vit.py --log --exp_id vit_l2_b128_adamw     --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer adamw
python run_exp_vit.py --log --exp_id vit_l2_b128_sgd       --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer sgd
python run_exp_vit.py --log --exp_id vit_l2_b128_momentum  --model vit --layers 2 --batch_size 128 --epochs 10 --optimizer momentum

python run_exp_vit.py --log --exp_id vit_l4_b128_adamw             --model vit --layers 4 --batch_size 128 --epochs 10 --optimizer adamw
python run_exp_vit.py --log --exp_id vit_l6_b128_adamw             --model vit --layers 6 --batch_size 128 --epochs 10 --optimizer adamw 
python run_exp_vit.py --log --exp_id vit_l6_b128_adamw_postnorm    --model vit --layers 6 --batch_size 128 --epochs 10 --optimizer adamw --block postnorm

