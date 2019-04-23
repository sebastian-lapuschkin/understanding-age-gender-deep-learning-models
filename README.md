# Understanding and Comparing Deep Neural Networks for Age and Gender Classification - Data and Models
This repository contains all the evaluated models for which results are reported in the ICCV 2017 workshop paper titled ["Understanding and Comparing Deep Neural Networks for Age and Gender Classification"](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Lapuschkin_Understanding_and_Comparing_ICCV_2017_paper.pdf)

[ ] TODO SPLASH IMAGE
[ ] TODO info about heatmap computation in paper.

That is, this repo contains the `deploy.prototxt` and `train_val.prototxt` files for all model architectures, pretraining and preprocessing choices for which performance measures are reported in the paper linked above.
`mean.binaryproto` files for outright deploying the models using Caffe are supplied as well.

Due to github's hard file size limit of 100mb per file, all model weights (i.e. the `.caffemodel` files) and `lmdb` data files are hosted externally (see section [Content] below).

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
- [ ] links to pretrained model weights as startingpoints
- [ ] links to lmdb files
- [ ] links to caffemodel files

# [] TODO RESULTS AND LINKS
