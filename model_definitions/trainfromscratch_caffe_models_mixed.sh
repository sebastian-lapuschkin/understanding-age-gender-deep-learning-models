#!/bin/bash

modelbackup=/media/lapuschkin/Data/FaceRecognitionModels
here=$PWD
flags=$here/FromScratchProgressFlags-$(date +%F)
mkdir $flags

#caffe reference gender models.
this=gender_fromscratch_caffereference_mixed/
cd $this
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/gender_cafferef_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/gender_cafferef_$i.done
done
mkdir $modelbackup/$this && mv -v models_test_is* $modelbackup/$this && cd $modelbackup/$this  && cd $modelbackup && python create_missing_symlinks.py && date >> $flags/gender_cafferef_symlinks.done
#go back up
cd $here



#caffe reference age models.
this=age_fromscratch_caffereference_mixed/
cd $this
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/age_cafferef_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/age_cafferef_$i.done
done
mkdir $modelbackup/$this && mv -v models_test_is* $modelbackup/$this && cd $modelbackup/$this  && cd $modelbackup && python create_missing_symlinks.py && date >> $flags/age_cafferef_symlinks.done
#go back up
cd $here










#googlenet gender models.
this=gender_fromscratch_googlenet_mixed/
cd $this
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/gender_googlenet_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/gender_googlenet_$i.done
done
mkdir $modelbackup/$this && mv -v models_test_is* $modelbackup/$this && cd $modelbackup/$this  && cd $modelbackup && python create_missing_symlinks.py && date >> $flags/gender_googlenet_symlinks.done
cd $here



#googlenet age models.
this=age_fromscratch_googlenet_mixed/
cd $this
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/age_googlenet_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/age_googlenet_$i.done
done
mkdir $modelbackup/$this && mv -v models_test_is* $modelbackup/$this && cd $modelbackup/$this  && cd $modelbackup && python create_missing_symlinks.py && date >> $flags/age_googlenet_symlinks.done

cd $here
echo "training script terminated."
