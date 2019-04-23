#!/bin/bash

#create necessary folders, since caffe tools do not create paths recursively
echo ''
echo 'Creating Folders'
echo ''
mkdir lmdb_mixed
mkdir mean_image_mixed
for i in {0..4}
do
	mkdir lmdb_mixed/Test_fold_is_$i
	mkdir mean_image_mixed/Test_fold_is_$i
done

echo ''
echo 'Running LMDB file creation'
echo ''
#run lmdb creation scripts
find ./*/ -name "create_lmdb*mixed.sh" -exec bash '{}' \;


echo ''
echo 'Running MEAN file creation'
echo ''
#compute mean files
find ./*/ -name "make_mean*mixed.sh" -exec bash '{}' \;

echo 'Done'
