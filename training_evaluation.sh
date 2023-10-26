### Bash script for training and evaluating a model. ###

#!/bin/bash

# Set the input and output directories

### --- DDD17 --- ###
EXP_DIR=/mnt/Seagate/DDD17/ddd17/model
TRAIN_DIR=/mnt/Seagate/DDD17/ddd17/model/data/training
VAL_DIR=/mnt/Seagate/DDD17/ddd17/model/data/validation
TEST_DIR=/mnt/Seagate/DDD17/ddd17/model/data/testing

### -- DDD20 --- ###
#EXP_DIR=/mnt/Seagate/DDD17/ddd20/model
#TRAIN_DIR=/mnt/Seagate/DDD17/ddd20/model/data/training
#VAL_DIR=/mnt/Seagate/DDD17/ddd20/model/data/validation
#TEST_DIR=/mnt/Seagate/DDD17/ddd20/model/data/testing

# --- Train and evaluate models for the different types of frames ---
echo -e "\n\n### Training the model for DVS Frames...###\n"
python3.6 ./cnn.py --frame_mode dvs --epochs 200 --experiment_rootdir $EXP_DIR/dvs --train_dir $TRAIN_DIR --val_dir $VAL_DIR
echo -e "\n\n### Evaluating the model for DVS Frames...###\n"
python3.6 ./evaluation.py --frame_mode dvs --experiment_rootdir $EXP_DIR/dvs --test_dir $TEST_DIR --weights_fname best_weights.h5
cp $EXP_DIR/data/scalers_dict.json $EXP_DIR/dvs
cp $EXP_DIR/data/percentiles.txt $EXP_DIR/dvs

echo -e "\n\n### Training the model for APS Frames...###\n"
python3.6 ./cnn.py --frame_mode aps --epochs 200 --experiment_rootdir $EXP_DIR/aps --train_dir $TRAIN_DIR --val_dir $VAL_DIR
echo -e "\n\n### Evaluating the model for APS Frames...###\n"
python3.6 ./evaluation.py --frame_mode aps --experiment_rootdir $EXP_DIR/aps --test_dir $TEST_DIR --weights_fname best_weights.h5
cp $EXP_DIR/data/scalers_dict.json $EXP_DIR/aps
cp $EXP_DIR/data/percentiles.txt $EXP_DIR/aps

echo -e "\n\n ### -------------------------------------------------- ### \n\n"
echo -e " Done training and evaluating models..."
echo -e "\n\n ### -------------------------------------------------- ### \n\n"
