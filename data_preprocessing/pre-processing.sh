### Bash script for Pre-processing. ###


#!/bin/bash
set -e

# Set the input and output directories

### --- DDD17 --- ###
ORIGIN_DIR=/mnt/Seagate/DDD17/ddd17/dataset
OUT_DIR=/mnt/Seagate/DDD17/ddd17/model/data
TODO_DIRS=(run2 run3 run4 run5)
SPLIT_DIR=/mnt/Seagate/DDD17/ddd17/not_used_ddd17/frames_too_long
TODO_FILES=(rec1487417411 rec1487424147 rec1487430438 rec1487594667 rec1487600962 rec1487846842 rec1487849663
      rec1487356509)


### --- FUSION --- ###
#OUT_DIR=/mnt/Seagate/DDD17/fusion/data


### --- DDD20 --- ###
#ORIGIN_DIR=/mnt/Seagate/DDD17/ddd20/dataset
##ORIGIN_DIR=/mnt/Seagate/DDD17/ddd20/not_used_ddd20/use_seperetaly/unpack_seperetely
##OUT_DIR=/mnt/Seagate/DDD17/ddd20/model/data
#TODO_DIRS=(aug01 aug02 aug04 aug05 aug06 aug08 aug09 aug12 aug13 aug14 aug15 jul01 jul02 jul05 jul08 jul09 jul16 jul17
#      jul18 jul20 jul23 jul24 jul27 jul28 jul29 jun25 jun28 jun29 jun30)
#SPLIT_DIR=/mnt/Seagate/DDD17/ddd20/not_used_ddd20/frames_too_long
#TODO_FILES=(rec1500399102 rec1500336443 rec1500831085 rec1501289546 rec1502648574 rec1500304763 rec1501191354
#      rec1500403661 rec1500331615 rec1500217992 rec1501189096 rec1498743713)
# The following files raise some problem while pre-processing :
# Buffer overflow error shows up when reduce_to_frame_based.py runs : aug15/rec1502838012.hdf5
# Can't be split : rec1502838012 (maybe also rec1502648574)
# Won't compute the percentiles : rec15003991021.hdf5 (it's the first split file of rec1500399102)


### --- PRE-PROCESSING --- ###

for TODO_DIR in "${TODO_DIRS[@]}"
do
  INPUT_DIR=${ORIGIN_DIR}/${TODO_DIR}
  BASE_ID=`basename ${INPUT_DIR}`
# Unpack data
  python2.7 ./reduce_to_frame_based.py --source_folder ${INPUT_DIR} --dest_folder ${OUT_DIR} --export_aps 1 --export_dvs 1
done

# Split HDF5 files larger than 2.7GB into two new ones
echo -e "\n### Splitting large HDF5 files...###"
for TODO_FILE in "${TODO_FILES[@]}"
do
  mkdir $SPLIT_DIR/$TODO_FILE
  mv $OUT_DIR/$TODO_FILE.hdf5 $SPLIT_DIR/$TODO_FILE/$TODO_FILE.hdf5
  python2.7 ./split_hdf5_files.py --parent_folder ${SPLIT_DIR} --in_file ${TODO_FILE}
  mv $SPLIT_DIR/$TODO_FILE/split/${TODO_FILE}1.hdf5 $OUT_DIR
  mv $SPLIT_DIR/$TODO_FILE/split/${TODO_FILE}2.hdf5 $OUT_DIR
done
# Only for DDD20
#rm $OUT_DIR/rec15003991021.hdf5
echo -e "\n### Done splitting...###\n"

# Split recordings
echo -e "\n\n### Splitting recordings... ###"
python2.7 ./split_recordings.py --source_folder ${OUT_DIR} --rewrite 1  --train_length 40 --test_length 20

# Compute percentiles
echo -e "\n\n### Computing percentiles... ###"
python2.7 ./compute_percentiles.py --source_folder ${OUT_DIR} --inf_pos_percentile 0.0 --sup_pos_percentile 0.9998 --inf_neg_percentile 0.0 --sup_neg_percentile 0.9998

# Export data and frames
echo -e "\n\n### Exporting data and frames... ###"
python2.7 ./export_cnn_data.py --source_folder ${OUT_DIR} --rotate 1

# Create validation set from training set
echo -e "\n\n### Creating validation set... ###"
python2.7 ./create_val_set.py --source_folder ${OUT_DIR} --percentage_split 0.1
