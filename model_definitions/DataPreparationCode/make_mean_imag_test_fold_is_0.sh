TOOLS=/home/lapuschkin/code/caffe/build/tools
DATA=/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning/lmdb/Test_fold_is_0/gender_train_lmdb
OUT=/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning/mean_image/Test_fold_is_0

$TOOLS/compute_image_mean.bin $DATA $OUT/mean.binaryproto

