print 'import: matplotlib.pyplot';  import matplotlib.pyplot as plt
print 'import: numpy';              import numpy as np
print 'import: caffe';              import caffe
print 'import: caffe.proto';        from caffe.proto import caffe_pb2
print 'import: lmdb';               import lmdb
print 'import: time';               import time
print 'import: json';               import json
print 'import: os';                 import os


def score_model(lmdb_path,meanimg_path,prototxt_path,modelweight_path,inputsize,useQuadro=False,useCPU=False):

    #create classifier
    print 'loading/parsing mean file'
    mean = caffe.io.caffe_pb2.BlobProto.FromString(open(meanimg_path,'rb').read())
    mean = caffe.io.blobproto_to_array(mean)[0]

    mean_img = np.transpose(mean,(1,2,0)) #caffeblob to brg from (dim, row, col) to (row,col,dim)
    mean_img = mean_img[...,::-1] #bgr to rgb

    print 'creating model'
    caffe.set_mode_gpu() #considerably faster
    device = 'Titan X'

    if useQuadro: caffe.set_device(1); device = 'Quadro' # selects Quadro, listed as index 0 in nvidia-smi? wtf
    if useCPU: caffe.set_mode_cpu(); device = 'CPU'

    net = caffe.Net(prototxt_path, modelweight_path, caffe.TEST )
    net.blobs['data'].reshape(1,3,inputsize,inputsize) # do once. important for predictor? blob size changes later

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_mean('data', mean) #disabled and done manually
    transformer.set_transpose('data',[2,0,1])
    transformer.set_channel_swap('data',[2,1,0])
    #transformer.set_raw_scale('data',255.) #this broke the prediction.




    #open lmdb file
    print 'opening lmdb file'
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    lmdb_size = lmdb_txn.stat()['entries']

    hit=0.0;                hit1off = 0.0;              confmat = None ;
    oversample_hit=0.0 ;    oversample_hit1off = 0.0;   oversample_confmat = None ;

    singletime=0.0 ;        oversample_time = 0.0
    count=0.0

    t_status_start = time.time()
    #read lmdb filen entries
    print 'reading lmdb file entries'
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)

        label = datum.label
        im = caffe.io.datum_to_array(datum).astype(np.float64) # type change for manual mean subtraction
        im = np.transpose(im,(1,2,0)) # from (dim, row, col) to (row,col,dim) #for visualization
        im = im[...,::-1] # from BGR to RGB #for visualization

        im -= mean_img #subtract mean image manually
        imshape = im.shape ; shapediff = (imshape[0]-inputsize, imshape[0]-inputsize)
        im_center = im[shapediff[0]:shapediff[0]+inputsize,shapediff[1]:shapediff[1]+inputsize,:] #compute center crop

        single_start = time.time()
        net.blobs['data'].reshape(1,3,inputsize,inputsize) #reset buffer size, since we also do oversampling evaluation
        net.blobs['data'].data[...] = transformer.preprocess('data',im_center)

        #standard one-sample-prediction
        output = net.forward()

        if inputsize == 227: # Adience or BVLC reference net
            prediction = output['prob'].argmax()
            if confmat is None : confmat = np.zeros([output['prob'].shape[1]]*2, dtype=int)
        elif inputsize == 224 and 'googlenet' in prototxt_path: # googlenet
            prediction = output['loss3/loss3'].argmax()
            if confmat is None : confmat = np.zeros([output['loss3/loss3'].shape[1]]*2, dtype=int)
        elif inputsize == 224 and 'vgg' in prototxt_path: #vgg16
            prediction = output['prob'].argmax()
            if confmat is None : confmat = np.zeros([output['prob'].shape[1]]*2, dtype=int)


        hit += (label == prediction)
        hit1off += (np.abs(label - prediction) <= 1)
        confmat[prediction,label] += 1
        singletime += time.time() - single_start


        #oversampling prediction
        oversampling_start = time.time()

        net.blobs['data'].reshape(10,3,inputsize,inputsize)
        ims = caffe.io.oversample([im],(inputsize,inputsize)) #manual mean subtraction here
        for i in xrange(len(ims)): #probably batch-executable somehow.
            net.blobs['data'].data[i,...] = transformer.preprocess('data',ims[i])


        oversample_outputs = net.forward()
        if inputsize == 227:
            oversample_prediction = oversample_outputs['prob'].mean(axis=0).argmax()
            if oversample_confmat is None : oversample_confmat = np.zeros([output['prob'].shape[1]]*2, dtype=int)
        elif inputsize == 224 and 'googlenet' in prototxt_path: # googlenet
            oversample_prediction = oversample_outputs['loss3/loss3'].mean(axis=0).argmax()
            if oversample_confmat is None : oversample_confmat = np.zeros([output['loss3/loss3'].shape[1]]*2, dtype=int)
        elif inputsize == 224 and 'vgg' in prototxt_path: # vgg16
            oversample_prediction = oversample_outputs['prob'].mean(axis=0).argmax()
            if oversample_confmat is None : oversample_confmat = np.zeros([output['prob'].shape[1]]*2, dtype=int)


        #fold0: {"acc": 0.54633920296570904, "acco": 0.57831325301204817, "1off": 0.88484708063021311, "1offo": 0.90523632993512515}
        oversample_hit += (label == oversample_prediction)
        oversample_hit1off += (np.abs(label - oversample_prediction) <= 1)
        oversample_confmat[oversample_prediction,label] += 1
        oversample_time += time.time() - oversampling_start

        count += 1 #count number of evaluated samples

        #print status every 100 and with the last sample
        n_status = 20
        if count % n_status == 0 or count == lmdb_size:
            t_status_end = time.time()
            print 'Evaluated model performance after {}/{} samples:'.format(int(count),lmdb_size)
            print 'SINGLE -- ACC: {}, 1-OFF: {} ({} s/image)'.format(np.round(100 * hit / count,2),np.round(100* hit1off / count,2), np.round(singletime/count,4))
            print 'OVRSMP -- ACC: {}, 1-OFF: {} ({} s/image)'.format(np.round(100 * oversample_hit / count,2),np.round(100* oversample_hit1off / count,2),np.round(oversample_time/count,4))
            print 'Total batch time for the last {} images: {} s ({})'.format(n_status,t_status_end - t_status_start,device)
            t_status_start = time.time()

        #if count > 100 : break #debugging

    return {'acc': hit/count, '1off': hit1off/count, 'acco':oversample_hit/count, '1offo':oversample_hit1off/count} , {'confmat': confmat.tolist(), 'confmato': oversample_confmat.tolist()}







''' MAIN '''


PATHPREFIX = '/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning'
PROBLEMS = ['age','gender']
MODELS = ['net_definitions','finetuning_caffereference','fromscratch_caffereference','finetuning_googlenet','fromscratch_googlenet', 'finetuning_vgg16', 'imdbwiki_vgg16']

MODELPOSTFIX = '_mixed' # '', _unaligned or _mixed
LMDBPOSTFIX = '_unaligned' # '', _unaligned
TESTSET = 'test' #train,test or val

#custom bitsm for the imdbwiki-trained googlenet:
MODELPATHPREFIX = '/home/lapuschkin/Desktop/to:woj'
#PROBLEMS = ['age']
MODELS = ['imdbwiki_googlenet']
#MODELS = ['finetuning_googlenet']


USE_QUADRO = False
USE_CPU = False
SCORE_ALL_MODEL_SNAPSHOTS = True
OVERWRITE = False

MODELS = [m+MODELPOSTFIX for m in MODELS]
for ageorgender in PROBLEMS:
    for model in MODELS:

        if 'googlenet' in model:
            inputsize = 224
            modeliteration = 170000
            snapshotsteps = 1000

            if 'imdbwiki' in model:
                modeliteration = 100000

        elif 'vgg' in model:
            inputsize = 224
            modeliteration = 125000
            snapshotsteps = 2500
        else:
            inputsize = 227
            modeliteration = 50000
            snapshotsteps = 1000

        if MODELPOSTFIX == '_mixed':
            modeliteration *= 2

        if SCORE_ALL_MODEL_SNAPSHOTS:
            nummodels = modeliteration/snapshotsteps
            MODELITERATIONS = (np.arange(1,nummodels+1)*snapshotsteps).tolist()
            #MODELITERATIONS = (np.arange(1,20)*50).tolist() # report results for model snapshopts from 50 to 950 in steps of 50
            #MODELITERATIONS = (np.arange(1,25)*100).tolist() # 100 to 2400 in steps of 1000
        else:
            MODELITERATIONS = [modeliteration]
            #MODELITERATIONS = [540]


        for modeliteration in MODELITERATIONS:
            scorefile = './results-{4}-vs-{3}/{0}_{1}_{2}.txt'.format(ageorgender,model,modeliteration,LMDBPOSTFIX,MODELPOSTFIX)

            #check whether the file exists and is complete:
            if os.path.isfile(scorefile):
                with open(scorefile, 'rb') as f:
                    content = f.read().split('\n')
                    if len(content) > 4 and content[-1].startswith('all:') and not OVERWRITE:
                        print scorefile, 'complete. skipping:'
                        print '\n'.join(content) + '\n'
                        continue #skip and dont overwrite existing results.
                    else:
                       print scorefile, 'needs work:', '(OVERWRITE)' if OVERWRITE else ''
                       print '\n'.join(content) + '\n'
                       time.sleep(5)


            try:
                if not os.path.isdir(os.path.dirname(scorefile)): os.mkdir(os.path.dirname(scorefile))

                with open(scorefile,'wb') as fh:
                    scoresoverfolds = []
                    confmatsoverfolds = []
                    for testfold in xrange(5):

                        lmdb_path = PATHPREFIX + '/lmdb{3}/Test_fold_is_{0}/{1}_{2}_lmdb'.format(testfold,ageorgender,TESTSET,LMDBPOSTFIX)
                        meanimg_path = PATHPREFIX + '/mean_image{1}/Test_fold_is_{0}/mean.binaryproto'.format(testfold,MODELPOSTFIX)
                        prototxt_path = MODELPATHPREFIX + '/{}_{}/deploy.prototxt'.format(ageorgender,model)
                        modelweight_path = MODELPATHPREFIX + '/{}_{}/models_test_is_{}/caffenet_train_iter_{}.caffemodel'.format(ageorgender,model,testfold,modeliteration)
                        #prototxt_path = PATHPREFIX + '/{}_{}/deploy.prototxt'.format(ageorgender,model)
                        #modelweight_path = PATHPREFIX + '/{}_{}/models_test_is_{}/caffenet_train_iter_{}.caffemodel'.format(ageorgender,model,testfold,modeliteration)

                        tstart=time.time()
                        print ''
                        print 'testing {}_{} iter{} fold {} vs lmdb{}'.format(ageorgender,model,modeliteration,testfold,LMDBPOSTFIX)
                        scores, confmats = score_model(lmdb_path,meanimg_path,prototxt_path,modelweight_path,inputsize,useQuadro=USE_QUADRO,useCPU=USE_CPU)
                        scoresoverfolds.append(scores)
                        confmatsoverfolds.append(confmats)
                        fh.write('fold{}: '.format(testfold) + json.dumps(scores) + ' ConfMats: ' + json.dumps(confmats) +'\n')
                        fh.flush()
                        tend=time.time()
                        print 'fold {} testing done after'.format(testfold), tend - tstart, 'seconds'

                    #write out averaged statistics
                    combined_scores = {}
                    combined_confmats = {}
                    keyset_scores = scoresoverfolds[0].keys()
                    keyset_confmats = confmatsoverfolds[0].keys()
                    for k in keyset_scores:
                        combined_scores[k] = str(np.round(np.mean([scoresoverfolds[i][k] for i in xrange(5)]),3))\
                                    + ' +- '\
                                    + str(np.round(np.std([scoresoverfolds[i][k] for i in xrange(5)]),3))

                    for k in keyset_confmats:
                        combined_confmats[k] = 0
                        for i in xrange(5): combined_confmats[k] += np.array(confmatsoverfolds[i][k],dtype=int)
                        combined_confmats[k] = combined_confmats[k].tolist()

                    fh.write('all: ' + json.dumps(combined_scores) + ' ConfMats: ' + json.dumps(combined_confmats))

            except Exception as e:
                print ''
                print 'ERROR WITH SCORE FILE', scorefile
                print e
                print ''
                time.sleep(10)

            #END for modeliteration in MODELITERATIONS
