### Bash file for creating images and videos with real and predicted values. ###

#!/bin/bash

# Set the input and output directories

### --- DDD17 --- ###
#EXP_DIR=/mnt/seagate/DDD17/ddd17/model
#TRAIN_DIR=/mnt/seagate/DDD17/ddd17/model/data/training
#VAL_DIR=/mnt/seagate/DDD17/ddd17/model/data/validation
#TEST_DIR=/mnt/seagate/DDD17/ddd17/model/data/testing
#VIDEO=rec14873330010

### --- DDD20 --- ###
EXP_DIR=/mnt/seagate/DDD17/ddd20/model
TRAIN_DIR=/mnt/seagate/DDD17/ddd20/model/data/training
VAL_DIR=/mnt/seagate/DDD17/ddd20/model/data/validation
TEST_DIR=/mnt/seagate/DDD17/ddd20/model/data/testing
VIDEO=rec14983556041

### --- DVS Frames --- ###
echo -e "\n### Copying necessary files...###"
mkdir -p $EXP_DIR/dvs/videos/process_videos/$VIDEO
cp -r $TEST_DIR/$VIDEO/* $EXP_DIR/dvs/videos/process_videos/$VIDEO
cp $EXP_DIR/dvs/percentiles.txt $EXP_DIR/dvs/videos
cp $EXP_DIR/dvs/scalers_dict.json $EXP_DIR/dvs/videos
echo -e "\n\n### Process new DVS video... ###"
python3.6 ./process_new_video.py --frame_mode dvs --experiment_rootdir $EXP_DIR/dvs --test_dir $TEST_DIR --video $VIDEO
echo -e "Done processing new video...\n\n"
echo -e "\n\n### Create images with results on new video... ###"
python3.6 ./viewer.py --frame_mode dvs --experiment_rootdir $EXP_DIR/dvs --test_dir $TEST_DIR --video_dir $EXP_DIR/dvs/videos/process_videos/$VIDEO
echo -e "Video and images created...\n\n"

### --- APS Frames --- ###
#echo -e "\n### Copying necessary files...###"
#mkdir -p $EXP_DIR/aps/videos/process_videos/$VIDEO
#cp -r $TEST_DIR/$VIDEO/* $EXP_DIR/aps/videos/process_videos/$VIDEO
#cp $EXP_DIR/aps/percentiles.txt $EXP_DIR/aps/videos
#cp $EXP_DIR/aps/scalers_dict.json $EXP_DIR/aps/videos
#echo -e "\n\n### Process new APS video... ###"
#python3.6 ./process_new_video.py --frame_mode aps --experiment_rootdir $EXP_DIR/aps --test_dir $TEST_DIR --video $VIDEO
#echo -e "Done processing new video...\n\n"
#echo -e "\n\n### Create images with results on new video... ###"
#python3.6 ./viewer.py --frame_mode aps --experiment_rootdir $EXP_DIR/aps --test_dir $TEST_DIR --video_dir $EXP_DIR/aps/videos/process_videos/$VIDEO
#echo -e "Video and images created...\n\n"
