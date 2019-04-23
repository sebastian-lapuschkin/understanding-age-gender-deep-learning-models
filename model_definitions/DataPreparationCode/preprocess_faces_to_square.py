import os
import numpy as np
import skimage.io

DATA_DIR='/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning/DATA/faces'
OUT_DIR='/home/lapuschkin/Desktop/FaceRecognition/code/AgeGenderDeepLearning/DATA/faces_squared'

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

for root, dirs, files in os.walk(DATA_DIR):

    if not '@' in root:
        continue

    foldername = os.path.basename(root)
    outfolder = OUT_DIR + '/' + foldername
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    for f in files:
        if not f.endswith('.jpg'):
            continue

        sourcefile = root + '/' + f
        destfile = outfolder + '/' + f

        img = skimage.io.imread(sourcefile)
        HWD = img.shape

        #pad the image.
        if HWD[0] > HWD[1]: # higher than wide
            diff = HWD[0] - HWD[1]
            leftadd = diff/2
            rightadd = diff - leftadd

            leftadd = np.repeat(img[:,0:1,:],leftadd,axis=1)
            rightadd = np.repeat(img[:,-1:: ,:],rightadd,axis=1)
            img = np.concatenate([leftadd,img,rightadd],axis=1)
        elif HWD[0] < HWD[1]: #wider than high
            diff = HWD[1] - HWD[0]
            topadd = diff/2
            bottomadd = diff - topadd

            topadd = np.repeat(img[0:1,...],topadd, axis=0)
            bottomadd = np.repeat(img[-1::,...],bottomadd,axis=0)
            img = np.concatenate([topadd,img,bottomadd],axis = 0)

        #img should be square now. (preserves aspect ratios by padding)
        #save image
        skimage.io.imsave(destfile,img)
        print 'processed:', foldername + '/' + f, HWD, '->', img.shape




    #for f in files:
    #    print f
