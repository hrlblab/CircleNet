## CircleNet-HG
cd ../src

# Train
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../models/ctdet_coco_hg.pth

# Evaluate AP / AR
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4 --dataset monuseg --load_model ../exp/circledet/CircleNet_HG_Reproduce/model_best.pth --test --ontestdata --debug 4
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.487
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.856
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.509
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.499
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.337
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.577
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.577
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.577
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.576
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.586
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000

## Rotation Test (Table 4, row 3)
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model ../exp/circledet/CircleNet_HG_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8918255489424661
