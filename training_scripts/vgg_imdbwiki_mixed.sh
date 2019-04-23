#!/bin/bash

modelbackup=/media/lapuschkin/Data/FaceRecognitionModels
here=$PWD
flags=$here/IMDBWIKI-mixed-FinetuningProgressFlags-$(date +%F)
mkdir $flags

caffe reference age models.
this=gender_imdbwiki_vgg16_mixed/
cd $this
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/gender_vgg16_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt --weights ./startingpoint/dex_imdb_wiki.caffemodel && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/gender_vgg16_$i.done
    mkdir $modelbackup/$this ;
    mv -v models_test_is_$i $modelbackup/$this
done
#cd $modelbackup/$this && cd $modelbackup && python create_missing_symlinks.py && date >> $flags/gender_vgg16_symlinks.done

#
#go back up
cd $here


#caffe reference age models.
this=age_imdbwiki_vgg16_mixed/
cd $this
for i in {0..4}
do
     #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/age_vgg16_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt --weights ./startingpoint/dex_imdb_wiki.caffemodel && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/age_vgg16_$i.done
    mkdir $modelbackup/$this
    mv -v models_test_is_$i $modelbackup/$this
done
#mkdir $modelbackup/$this ; mv -v models_test_is* $modelbackup/$this && cd $modelbackup/$this && cd $modelbackup && python create_missing_symlinks.py && date >> $flags/age_vgg16_symlinks.done

#go back up
#cd $here



echo "training script terminated."

