TOOLS=/home/lapuschkin/code/caffe/build/tools
DATA=/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning/lmdb_mixed/Test_fold_is_1/gender_train_lmdb
OUT=/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning/mean_image_mixed/Test_fold_is_1

$TOOLS/compute_image_mean.bin $DATA $OUT/mean.binaryproto

