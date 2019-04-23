#!/bin/bash
#this script replaces all generic /home/ubuntu/AdienceFaces paths to he current one in all scripts.
# I used this script after checking out https://github.com/GilLevi/AgeGenderDeepLearning
# You might find it useful, after adapting some of the paths


#TODO: adjust this to where your caffe is
CAFFEPATH=$HOME/code/caffe

echo ''
echo 'Fixing Project Root Paths'
echo ''
#change root paths
find ./*/ -type f -exec sed -i -e s^/home/ubuntu/AdienceFaces^$PWD^g '{}' \;

echo ''
echo 'Setting Caffe Tool Path'
echo ''
#change caffe tools path 
find ./*/ -type f -exec sed -i -e s^/home/ubuntu/repositories/caffe^$CAFFEPATH^g '{}' \;

echo ''
echo 'Unifying LMDB and MEAN file naming patterns'
echo ''
#unify path definitions for lmdb and mean_images
find ./*/ -type f -exec sed -i -e s^/mean_image/Test_folder_is_^/mean_image/Test_fold_is_^g '{}' \;
