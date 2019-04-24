import os
import subprocess

def write_hm_config(cfgpath, cfgvals):
    """ writes a config file for the demonstrator code.
        require some values in cfgvals:
        deploy.prototxt : path to that file
        weights : path to the model weights
        meanfile : path to the mean file (binary blob)
        synsetfile : path to the problem specific synset file
        numclasses : the number of classes to classify into
        baseimgsize : the input dimensions (one axis) for the model
        stoplayer : the highest index of the layers at the bottom. you know what.
    """

    lines = ['param_file',\
             cfgvals['deploy.prototxt'],\
             '',\
             'model_file',\
             cfgvals['weights'],\
             '',\
             'mean_file',\
             cfgvals['meanfile'],\
             '',\
             'synsetfile',\
             cfgvals['synsetfile'],\
             '',\
             'use_mean_file_asbinaryprotoblob',\
             '1',\
             '',\
             'lastlayerindex',\
             '-2',\
             '',\
             'firstlayerindex',\
             '0',\
             '',\
             'numclasses',\
             cfgvals['numclasses'],\
             '',\
             'baseimgsize',\
             cfgvals['baseimgsize'],\
             '',\
             'standalone_outpath',\
             cfgvals['outpath'],\
             '',\
             'standalone_rootpath',\
             'images',\
             '',\
             'epsstab',\
             '0.1',\
             '',\
             'alphabeta_beta',\
             '1',\
             '',\
             'relpropformulatype',\
             cfgvals['relpropformulatype'],\
             '',\
             'auxiliaryvariable_maxlayerindexforflatdistinconv',\
             cfgvals['stoplayer'],\
             ''
            ]

    if not os.path.exists(os.path.dirname(cfgpath)): os.makedirs(os.path.dirname(cfgpath))
    with open(cfgpath, 'wb') as f:
        f.write('\n'.join(lines))



#lrp_bin = '/home/lapuschkin/Desktop/ComputeFaceHeatmaps/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_demo_minimal_output'
lrp_bin = '/home/lapuschkin/Desktop/ComputeFaceHeatmaps/lrp_toolbox/caffe-master-lrp/demonstrator/lrp_demo'
FRD = '/media/lapuschkin/Data/FaceRecognitionData'
FRM = '/media/lapuschkin/Data/FaceRecognitionModels'
AGDL = '/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning'

import glob


classtopredict = 0
experimentname = 'grandma-age-{}'.format(classtopredict)
experimentname = 'test-age-{}'.format(classtopredict)
PROBLEMSETS = ['age']
INIT = ['finetuning']
#MODELS = ['googlenet', 'caffereference', 'vgg16']
MODELS = ['vgg16']
PREPROP = ['mixed']

#classtopredict = -1
relpropformulatype = '102' #100, 102, 104 #104
# 100 = composite
# 102 = composite + flat
# 104 = composite + ww

images = ['{}r.png'.format(i) for i in [16, 20, 23]]
images = ['19r.png']
#images = glob.glob('images/*.png')

metascriptpath = './experiments/{}.sh'.format(experimentname)
with open(metascriptpath, 'wb') as metascript:

    for problem in PROBLEMSETS:
        for init in INIT: #['fromscratch', 'finetuning', 'imdbwiki']:
            for model in MODELS: #['googlenet', 'caffereference', 'vgg16']:
                scriptpath = './experiments/{}-{}-{}-{}.sh'.format(experimentname, problem, init, model)
                with open(scriptpath, 'w') as script:
                    for data in PREPROP: # ['aligned', 'unaligned', 'mixed']
                        for fold in xrange(5):

                            cfgpath = './experiments/{}/{}_{}_{}_{}/fold{}/config.txt'.format(experimentname, problem, init, model, data, fold)

                            cfgvals = {}

                            #relpropformulatype
                            cfgvals['relpropformulatype'] = relpropformulatype

                            #input sizes
                            if model == 'caffereference': cfgvals['baseimgsize'] = '227'
                            if model == 'googlenet' or model == 'vgg16': cfgvals['baseimgsize'] = '224'

                            #num classes
                            if problem == 'age': cfgvals['numclasses'] = '8'
                            if problem == 'gender': cfgvals['numclasses'] = '2'

                            #synset files
                            if problem == 'age': cfgvals['synsetfile'] = os.path.abspath('./synset_age.txt')
                            if problem == 'gender': cfgvals['synsetfile'] = os.path.abspath('./synset_gender.txt')

                            #mean files
                            if data == 'aligned': cfgvals['meanfile'] = FRD + '/mean_image/Test_fold_is_{}/mean.binaryproto'.format(fold)
                            if data == 'unaligned' or data == 'mixed': cfgvals['meanfile'] = FRD + '/mean_image_{}/Test_fold_is_{}/mean.binaryproto'.format(data, fold)

                            #deploy prototxt files:
                            if data == 'aligned': cfgvals['deploy.prototxt'] = AGDL + '/{}_{}_{}/deploy.prototxt'.format(problem, init, model)
                            if data == 'unaligned' or data == 'mixed': cfgvals['deploy.prototxt'] = AGDL + '/{}_{}_{}_{}/deploy.prototxt'.format(problem, init, model, data)

                            #weights
                            modeliter = -1
                            if model == 'caffereference': modeliter = 50000
                            if model == 'googlenet': modeliter = 170000
                            if model == 'vgg16': modeliter = 125000
                            if data == 'mixed': modeliter *= 2

                            if data == 'aligned': cfgvals['weights'] = FRM + '/{}_{}_{}/models_test_is_{}/caffenet_train_iter_{}.caffemodel'.format(problem, init, model, fold, modeliter)
                            if data == 'unaligned' or data == 'mixed': cfgvals['weights'] = FRM + '/{}_{}_{}_{}/models_test_is_{}/caffenet_train_iter_{}.caffemodel'.format(problem, init, model, data, fold, modeliter)

                            #outpath
                            cfgvals['outpath'] = './experiments/{}/{}_{}_{}_{}/fold{}'.format(experimentname, problem, init, model, data, fold)

                            if model == 'caffereference' : cfgvals['stoplayer'] = '0'
                            if model == 'googlenet': cfgvals['stoplayer'] = '4' # 4, 8, 10
                            if model == 'vgg16': cfgvals['stoplayer'] = '9' #?

                            #cfgvals: write them
                            write_hm_config(cfgpath, cfgvals)

                            #write testfilelist
                            testfilelistpath = cfgvals['outpath'] + '/testfilelist.txt' # folder exists after writing the config
                            with open(testfilelistpath,'wb') as f:
                                f.write('\n'.join(['images/{} {}'.format(i,classtopredict) for i in images]))

                            #write script to execute this heatmap computation
                            script.write(' '.join([lrp_bin, cfgpath, testfilelistpath, '.']) + '\n')
                            #add lines to run div. heatmap visualizations in parallel


                            hmoutfold = './heatmaps/{}/fAVG'.format(experimentname)
                            if not os.path.isdir(hmoutfold): os.makedirs(hmoutfold)


                            for i in images:
                                rawhmpath = '{}/images/{}_rawhm.txt'.format(os.path.dirname(cfgpath), i)
                                hmoutpath = './heatmaps/{}/{}/{}-{}_{}_{}_{}_i{}_f{}-class{}{} &'.format(experimentname, i, i, problem, init, model, data, modeliter, fold,classtopredict, '.png')
                                if not os.path.isdir(os.path.dirname(hmoutpath)): os.makedirs(os.path.dirname(hmoutpath))

                                script.write('python apply_heatmap.py {} black-firered {} \n'.format(rawhmpath,hmoutpath))

                                #compute the average heatmap over all folds and heatmap as well
                                if fold == 4:
                                    hmoutpath = './heatmaps/{}/fAVG/{}-{}_{}_{}_{}_i{}_fAVG-class{}{} &'.format(experimentname, i, problem, init, model, data, modeliter, classtopredict, '.png')
                                    script.write('python compute_average_heatmaps.py {} {} {}\n'.format(os.path.dirname(os.path.dirname(cfgpath)), i, hmoutpath))
                                #endif fold
                            #endif for i in images

                        #endfor folds
                    #endfor data

                    script.write('echo done!')
                    metascript.write('bash {} &\n'.format(scriptpath))
                    metascript.write('echo {} running.\n'.format(scriptpath))
                #endfor with open
            #endfor models
        #endfor init
    #endfor problem




#running meta-script
print 'done. now calling', 'bash', metascriptpath
subprocess.call(['bash', metascriptpath])










