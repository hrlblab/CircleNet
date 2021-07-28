# CircleNet-DLA
## Train
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/models/ctdet_coco_dla_2x.pth
## AP Test (Table 3, row 7)
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/circledet/CircleNet_DLA_Reproduce/model_best.pth --test --ontestdata --debug 4
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
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/circledet/CircleNet_DLA_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8855951478392722

# CircleNet-HG
## Train
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/models/ctdet_coco_hg.pth
## AP Test (Table 3, row 6)
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4 --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/circledet/CircleNet_HG_Reproduce/model_best.pth --test --ontestdata --debug 4
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
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/circledet/CircleNet_HG_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8918255489424661

# CenterNet-DLA
## Train
python main.py ctdet --exp_id CenterNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/models/ctdet_coco_dla_2x.pth
## AP Test
python main.py ctdet --exp_id CenterNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/ctdet/CenterNet_DLA_Reproduce/model_best.pth --test --ontestdata --debug 4
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

## Rotation Test (Table 4, row 2)
python main.py ctdet --exp_id CenterNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/ctdet/CenterNet_DLA_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8480562669622721

# CenterNet-HG
## Train
python main.py ctdet --exp_id CenterNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/models/ctdet_coco_hg.pth
## AP Test
python main.py ctdet --exp_id CenterNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/ctdet/CenterNet_HG_Reproduce/model_best.pth --test --ontestdata --debug 4
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

## Rotation Test (Table 4, row 1)
python main.py ctdet --exp_id CenterNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model /home/sybbure/CircleNet/CircleNet/exp/ctdet/CenterNet_HG_Reproduce/model_best.pth --test --ontestdata --rotate_reproduce 90
# 0.8207539105906487