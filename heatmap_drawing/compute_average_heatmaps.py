import sys
import numpy as np
import apply_heatmap

#methods and functions
def load_relevance_scores(heatmappath):
    if heatmappath.endswith('.npy') or heatmappath.endswith('.npz'):
        R = np.load(heatmappath)
    elif heatmappath.endswith('.mat'):
        import scipy.io as matio
        R = matio.loadmat
        # pick first "proper" variable
        for k in R.keys():
            if not k.startswith('__') and not k.startswith('MATLAB '):
                R = R[k]
                break
    elif heatmappath.endswith('.txt'):
        with open(heatmappath, 'rb') as f:
            #heuristically check for alex' format
            content = f.read().split('\n')
            line1 = [c for c in content[0].split(' ') if len(c) > 0]
            line2 = [c for c in content[1].split(' ') if len(c) > 0]
            if len(line1) == 1 and line1[0] == '3' and len(line2) == 2: # we do probably have an alex-formatted file.
                R = load_alex_format(content)
            else: # assume block matrix
                R = np.loadtxt(heatmappath)
    #we now have loaded some heatmaps. those may ha(R):ve relevance values for eac color channe, e.g. be H x W x 3 dimensional.
    #we flatten by summing over the color channels.
    #FLATTENING CAN LATER BE DONE PER COLOR MAPPING: THIS MIGHT BE USED FOR ACTUALLY MAPPING STUFF
    R = R.sum(axis=2)
    return R

def load_alex_format(lines):
    D = int(lines[0].split(' ')[0])
    H,W = [int(v) for v in lines[1].split(' ') if len(v) > 0]
    R = np.zeros([H,W,D])

    OFFSET = 2
    for d in xrange(D):
        for h in xrange(H):
            row = [float(v) for v in lines[d*H + h + OFFSET].split(' ') if len(v) > 0]
            R[h,:,d] = np.array(row)

    return R



this, folder, iname, outpath = sys.argv

R = 0 # init R
for f in xrange(5):

   hmpath = '{}/fold{}/images/{}_rawhm.txt'.format(folder,f,iname)
   R += load_relevance_scores(hmpath)


#compute the average
R /= 5.

RGB = apply_heatmap.apply_colormap(R,'black-firered')
apply_heatmap.write_heatmap_image(RGB,outpath)
