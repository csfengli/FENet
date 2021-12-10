## Requirements：
troch-encoding=1.2.2b20210508
pytorch=1.7.1
torchvision=0.8.2
mlflow=1.16.0
barbar=0.2.1
numpy=1.20.1
pillow=8.2.0

## Training
python main.py --dataset 'GTOS-mobile' --scheduler 'cosine' --model FENet --backbone 'Resnet18' --lr 0.007 --train_BS 64 --test_BS 64 --num_epochs 30 --save_name 1 --seed 2 --train_need

python main.py --dataset 'GTOS-mobile' --scheduler 'cosine' --model FENet --backbone 'Resnet50' --lr 0.008 --train_BS 64 --test_BS 64 --num_epochs 30 --save_name 1 --seed 2 --train_need

## Testing
python main.py --dataset 'GTOS-mobile' --scheduler 'cosine' --model FENet --backbone 'Resnet18' --lr 0.007 --train_BS 64 --test_BS 64 --num_epochs 30 --save_name 1 --seed 2 --resume --resume_path ./results/texture_recognition/GTOS-mobile/FENet_18_1/Best_Weights.pt

python main.py --dataset 'GTOS-mobile' --scheduler 'cosine' --model FENet --backbone 'Resnet50' --lr 0.008 --train_BS 64 --test_BS 64 --num_epochs 30 --save_name 1 --seed 2 --resume --resume_path ./results/texture_recognition/GTOS-mobile/FENet_50_1/Best_Weights.pt