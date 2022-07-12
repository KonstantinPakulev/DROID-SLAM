#!/bin/bash

SAM_PATH=/home/konstantin/patata/samsung_office_3m
ASSOC_PATH=/home/konstantin/patata/samsung_office_3m/associations.txt
OUTPUT_FILE=/home/konstantin/patata/samsung_office_3m/droid_slam_traj.txt
WEIGHTS_PATH=../droid.pth

python ../evaluation_scripts/test_sam_office.py --datapath=$SAM_PATH --association_file=$ASSOC_PATH --output_file=$OUTPUT_FILE --weights=$WEIGHTS_PATH --disable_vis

# Add for disabling visualization
#--disable_vis