#!/bin/bash

TRAIN_DIR=/home/eltwod/Thesis/Final_Code/model/training
VAL_DIR=/home/eltwod/Thesis/Final_Code/model/validation
TEST_DIR=/home/eltwod/Thesis/Final_Code/model/testing

N_exp=5
min_lr=0.00005
max_lr=0.0005

step=$(bc -l <<< "($max_lr - $min_lr) /($N_exp - 1)")


for lr in `seq $min_lr $step $max_lr`
do
	exp_rootdir=/home/eltwod/Thesis/Final_Code/model

	# Train
	python3.4 ../cnn.py --experiment_rootdir=$exp_rootdir \
 	--train_dir=$TRAIN_DIR --val_dir=$VAL_DIR --frame_mode=dvs \
	--initial_lr=$lr --epochs=30 --norandom_seed

	# Test
	python3.4 ../evaluation.py --experiment_rootdir=$exp_rootdir --test_dir=$TEST_DIR \
		--frame_mode=dvs --weights_fname=weights_001.h5

done