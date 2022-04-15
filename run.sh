#!/bin/sh
python seg_train.py --dataset short_axis_full --arch NestedUNet --loss BCEDiceLoss --name="short_axis" --epoch=30
