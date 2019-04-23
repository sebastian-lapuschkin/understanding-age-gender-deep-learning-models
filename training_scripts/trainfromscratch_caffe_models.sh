#!/bin/bash

here=$PWD
flags=$here/FromScratchProgressFlags-$(date +%F)
mkdir $flags

#caffe reference gender models.
cd gender_fromscratch_caffereference
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/gender_cafferef_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/gender_cafferef_$i.done
done

#go back up
cd $here



#caffe reference age models.
cd age_fromscratch_caffereference
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/age_cafferef_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/age_cafferef_$i.done
done

#go back up
cd $here










#googlenet gender models.
cd gender_fromscratch_googlenet
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/gender_googlenet_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/gender_googlenet_$i.done
done

cd $here



#googlenet age models.
cd age_fromscratch_googlenet
for i in {0..4}
do
    #create start flag. on suceess start training. on success move model files. on success create done flag.
    date >> $flags/age_googlenet_$i.started && caffe train --solver solver_test_fold_is_$i.prototxt && mkdir models_test_is_$i && mv *.caffemodel models_test_is_$i/ && mv *.solverstate models_test_is_$i/ && date >> $flags/age_googlenet_$i.done
done

cd $here
echo "training script terminated."
