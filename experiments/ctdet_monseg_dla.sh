# CenterNet-DLA
cd ../src
## Train
python main.py ctdet --exp_id CenterNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../models/ctdet_coco_dla_2x.pth
## AP Test
python main.py ctdet --exp_id CenterNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../exp/ctdet/CenterNet_DLA_Reproduce/model_best.pth --test --ontestdata --debug 4
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.399
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.826
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.315
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.403
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.338
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.491
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.491
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.491
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.489
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.604
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000

## Rotation Test
python main.py ctdet --exp_id CenterNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model ../exp/ctdet/CenterNet_DLA_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8480562669622721