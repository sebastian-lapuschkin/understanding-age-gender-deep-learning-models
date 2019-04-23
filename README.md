# Understanding and Comparing Deep Neural Networks for Age and Gender Classification - Data and Models
This repository contains all the evaluated models for which results are reported in the ICCV 2017 workshop paper titled ["Understanding and Comparing Deep Neural Networks for Age and Gender Classification"](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Lapuschkin_Understanding_and_Comparing_ICCV_2017_paper.pdf)

[ ] TODO SPLASH IMAGE
[ ] TODO info about heatmap computation in paper.

That is, this repo contains the `.caffemodel`, `deploy.prototxt` and `train_val.prototxt` files for all model architectures, pretraining and preprocessing choices for which performance measures are reported in the paper linked above.
`mean.binaryproto` files for outright deploying the models are supplied as well. [ ]TODO LINKS links to the `lmdb` data files used for training are linked below.

Pretrained model starting weights, for reproducing the results "from scratch" can be found in the data.

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

# [] TODO RESULTS AND LINKS
