#!/bin/bash
echo "HELLO WORLD"
CLASSIFIER='step7'
SOURCE_DIR=../data/cropped_image_data/$CLASSIFIER
TRAIN_DEST_DIR=../data/splitted_cropped_image_data/train/$CLASSIFIER
VAL_DEST_DIR=../data/splitted_cropped_image_data/val/$CLASSIFIER
splitt=0.8

rm -r $TRAIN_DEST_DIR
rm -r $VAL_DEST_DIR
mkdir $TRAIN_DEST_DIR
mkdir $VAL_DEST_DIR

echo   $TRAIN_DEST_DIR
#SOURCE_DIR=../test/
#TRAIN_DEST_DIR=../splittest/
#VAL_DEST_DIR=../valsplittest/
#dirs=$(find $SOURCE_DIR -mindepth 1 -maxdepth 1 -type d | xargs basename)
#for dir in "${dirs[@]}"
#do
#    echo "$dir"
#done


total_number_of_image=$(find $SOURCE_DIR -type f | wc -l) #get total number files in the folder
train_files=`echo $total_number_of_image*$splitt | bc` #calculate total files for training
train_files=${train_files%.*} #round it to integer number
val_files=`echo $total_number_of_image-$train_files |bc`

echo $train_files
echo $val_files

cp $(find $SOURCE_DIR -type f | head -$train_files) $TRAIN_DEST_DIR
cp $(find $SOURCE_DIR -type f | tail -$val_files) $VAL_DEST_DIR
#find $SOURCE_DIR -maxdepth 1 -type f |head -$train_files | xargs -I _ cp _  $TRAIN_DEST_DIR
#find $SOURCE_DIR -maxdepth 1 -type f |tail -$val_files | xargs -I _ cp _  "$VAL_DEST_DIR"
echo "DONE COPY FILE!!"