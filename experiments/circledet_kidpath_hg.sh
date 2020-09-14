cd ../src
# train
python main.py circledet --exp_id kidpath_hg --arch hourglass --batch_size 8 --master_batch 4 --lr 2.5e-4  --load_model ../models/ExtremeNet_500000.pth --gpus 1,2 --print_iter 1  --dataset kidpath
# test
python test.py circledet --exp_id kidpath_hg --arch hourglass --keep_res --resume --dataset kidpath
# flip test
python test.py circledet --exp_id kidpath_hg --arch hourglass --keep_res --resume --flip_test --dataset kidpath
# multi scale test
python test.py circledet --exp_id kidpath_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --dataset kidpath
cd ..