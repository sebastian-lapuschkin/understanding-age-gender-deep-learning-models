#!/bin/bash

#create necessary folders, since caffe tools do not create paths recursively
echo ''
echo 'Creating Folders'
echo ''
mkdir lmdb_unaligned
mkdir mean_image_unaligned
for i in {0..4}
do
	mkdir lmdb_unaligned/Test_fold_is_$i
	mkdir mean_image_unaligned/Test_fold_is_$i
done

echo ''
echo 'Running LMDB file creation'
echo ''
#run lmdb creation scripts
find ./*/ -name "create_lmdb*unaligned.sh" -exec bash '{}' \;


echo ''
echo 'Running MEAN file creation'
echo ''
#compute mean files
find ./*/ -name "make_mean*unaligned.sh" -exec bash '{}' \;

echo 'Done'
