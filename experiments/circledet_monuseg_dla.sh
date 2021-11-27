# CircleNet-DLA
cd ../src
## Train
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../models/ctdet_coco_dla_2x.pth
## AP Test (Table 3, row 7)
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../exp/circledet/CircleNet_DLA_Reproduce/model_best.pth --test --ontestdata --debug 4
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.486
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.855
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.516
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.499
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.305
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.578
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.578
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.578
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.577
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.621
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000

## Rotation Test (Table 4, row 4)
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model ../exp/circledet/CircleNet_DLA_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8855951478392722