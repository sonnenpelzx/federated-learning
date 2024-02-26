python3 main_fed_accuracy.py --dataset mnist --num_channels 1 --model cnn --epochs 100 --iid --gpu 0 --frac 0.2&
python3 main_fed_accuracy.py --dataset mnist --num_channels 1 --model cnn --epochs 100 --iid --gpu 0 --frac 0.4&
python3 main_fed_accuracy.py --dataset mnist --num_channels 1 --model cnn --epochs 100 --iid --gpu 0 --frac 0.6&
wait