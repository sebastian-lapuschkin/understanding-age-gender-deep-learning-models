# Understanding and Comparing Deep Neural Networks for Age and Gender Classification - Data and Models
This repository contains all the evaluated models for which results are reported in the ICCV 2017 workshop paper titled ["Understanding and Comparing Deep Neural Networks for Age and Gender Classification"](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Lapuschkin_Understanding_and_Comparing_ICCV_2017_paper.pdf)

[ ] TODO SPLASH IMAGE
[ ] TODO info about heatmap computation in paper.

That is, this repo contains the `deploy.prototxt` and `train_val.prototxt` files for all model architectures, pretraining and preprocessing choices for which performance measures are reported in the paper linked above.
`mean.binaryproto` files for outright deploying the models using Caffe are supplied as well.

Due to github's hard file size limit of 100mb per file, all model weights (i.e. the `*.caffemodel` files) and `lmdb` data files are hosted externally, via a nextcloud service of the Fraunhofer Heinrich Hertz Institute (see section [Content](https://github.com/sebastian-lapuschkin/understanding-age-gender-deep-learning-models/blob/master/README.md#content) below).

This repository can be understood as an extension of [Gil Levi's age and gender deep learning project page](https://github.com/GilLevi/AgeGenderDeepLearning), this page's paper originally found its foundation in.

Should you use any code or models from this github repository, please cite the corresponding paper:
```
@incproceedings{lapuschkin2017understanding,
  author = {Lapuschkin, Sebastian and Binder, Alexander and M\"uller, Klaus-Robert and Samek, Wojciech},
  title = {Understanding and Comparing Deep Neural Networks for Age and Gender Classification},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW)},
  pages = {1629-1638},
  year = {2017},
  doi = {10.1109/ICCVW.2017.191},
  url = {https://doi.org/10.1109/ICCVW.2017.191}
}
```

All heatmap visualizations shown in the paper, such as the image at the top of the page, have been generated using the LRP implementation for Caffe, as provided by in the [LRP Toolbox](https://github.com/sebastian-lapuschkin/lrp_toolbox).

## Content
- Folder `folds` contains the dataset split description for the [Adience benchmark data](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender) used for training and evaluation. This folder is an extension to the one found in [Gil Levi's repo](https://github.com/GilLevi/AgeGenderDeepLearning) and contains additional preprocessing settings.
- `training_scripts` contains shell scripts used for starting the training of the neural network models.
- `DataPrepartionCode` contains scripts for generating `mean.binaryproto` and `lmdb` binary blobs from raw Adience image data. This folder is an extension to the one found in [Gil Levi's repo](https://github.com/GilLevi/AgeGenderDeepLearning) and contains additional preprocessing settings.
- The folder `mean_images` contains the `mean.binaryproto` files for all folds and preprocessing choices, as used for training, validation and testing
- The folder `model_definitions` contains the `*.prototxt` files for Caffe, i.e. a description of the model architecture each. Here, a naming pattern `[target]_[init]_[arch][_preproc]` applies, where
  + `target` is from `{age, gender}` and describes the prediction problem
  + `init` is from `{fromscratch, finetuning, imdbwiki}` and describes random initialization, a weight intialization from ImageNet pretraining, and a weight initialization from ImageNet pretraining followed by [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) pretraining, respectively.
  + `arch` is from `{caffereference, googlenet, vgg16, net_definitions}` and describes the architecture of the model. Here, `net_definitions` refers to the model architecture used in [Gil Levi's repo](https://github.com/GilLevi/AgeGenderDeepLearning).
  + The `_preproc` suffix is optional and refers to `_unaligned` images (i.e. training images only under rotation alignment), aligned training images (landmark-based alignment, so suffix) or `_mixed` alignment, (i.e. both images under landmark-based and rotation-based alignment are used for trainng)
  + [ ] TODO add starting weights to datacloud
- The `lmdb`files used for model training, validation testing can be downloaded [here](https://datacloud.hhi.fraunhofer.de/nextcloud/s/n6BLLnGPzinbe55).
-  The model weights (i.e. the `*.caffemodel` files) to the neural network descriptions contained in this repository can be downloaded [here](https://datacloud.hhi.fraunhofer.de/nextcloud/s/TQnGNJmQZLWkQ7X). These files match the model definitions in folder `model_definitions`

**Note** that you will have to adapt the (absolute) paths denoted in scripts and model description files in order to use the code.

## Result overview
Below table briefly presents the obtained results from the paper this repository belongs to.
- [ ] TODO
