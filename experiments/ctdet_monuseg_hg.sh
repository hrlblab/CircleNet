# CenterNet-HG
cd ../src
## Train
python main.py ctdet --exp_id CenterNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../models/ctdet_coco_hg.pth
## AP Test
python main.py ctdet --exp_id CenterNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../exp/ctdet/CenterNet_HG_Reproduce/model_best.pth --test --ontestdata --debug 4
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.447
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.846
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.427
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.451
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.395
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.539
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.539
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.539
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.537
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.608
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000

## Rotation Test
python main.py ctdet --exp_id CenterNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model ../exp/ctdet/CenterNet_HG_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8207539105906487