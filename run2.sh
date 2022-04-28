#!/bin/sh
python vol_auto_validate.py --dataset formatted_kaggle_data_bowl --arch VolumeEstimation --loss RMSELoss --name [name] --epoch=50 --num_short=1 --use_long_axis=False --pooling_method average