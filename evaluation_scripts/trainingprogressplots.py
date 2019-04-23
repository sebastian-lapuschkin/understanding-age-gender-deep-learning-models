""" this script is used to create line plots showing the training performance over time """

# first, create a data structure to hold all the data.
# we create a dictionary <model> -> score sequence
# <model> here is described by a cascade of keys [ageorgender][weightinit][modeltype][trainingdata][testdata][score][modeliter]

import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import json
import cPickle


def scan_files():
    folders = glob.glob('./results-*')
    for folder in folders:
        print 'scanning',folder,'...'
        txtfiles = glob.glob(folder + '/*.txt')
        for txt in txtfiles:
            print txt
            with open(txt, 'rb') as f:
                content = f.read().split('\n')
                if not content[-1].startswith('all:') or not 'ConfMats:' in content[-1]:
                    print '    file',txt, 'incomplete?'
                    print '    ','\n'.join(content)
                    print ''
    print 'done.'

def scan_file(filename):
    with open(filename, 'rb') as f:
        content = f.read().split('\n')
        if not content[-1].startswith('all:') or not 'ConfMats:' in content[-1]:
            return False
        else:
            return True


def readResults(resultdict,filename):
    if not scan_file(filename):
        print '             skipping incomplete file', filename
        return


    #get all keys from the file name first.
    i = filename.rfind('results-') #find rightmost occurrence of this substring.
    if i < 0: raise ValueError('Invalid file or path name for: {}'.format(filename))

    validstring = filename[i::]
    keys = os.path.dirname(validstring).split('-') # get info about training and test data
    traindata = keys[1]
    testdata = keys[3]
    if traindata == '':
        traindata = 'aligned'
    else:
        traindata = traindata.lstrip('_') #remove leading '_'

    if testdata == '':
        testdata = 'aligned'
    else:
        testdata = testdata.lstrip('_')


    #get remaining keys. age/gender , weightinit, modeltype, modeliter
    keys = os.path.splitext(os.path.basename(validstring))[0].split('_')
    #print validstring
    #print keys
    if 'net_definitions' in validstring:
        #handle adience network files
        ageorgender = keys[0]
        weightinit = 'fromscratch'
        modeltype = 'adiencenet'
        modeliter = int(keys[-1])

    else:
        #regular files
        ageorgender = keys[0]
        weightinit = keys[1]
        modeltype = keys[2]
        modeliter = int(keys[-1])


    #read out fold-average accuracy values
    with open(filename,'rb') as f:
        avgresults = f.read().split('\n')[-1]
        avgresults = json.loads(avgresults.split('ConfMats')[0].split('all:')[1])

        acc = float(avgresults['acc'].split('+-')[0])
        acco = float(avgresults['acco'].split('+-')[0])
        off = float(avgresults['1off'].split('+-')[0])
        offo = float(avgresults['1offo'].split('+-')[0])


    # now we have to add the keys
    # ageorgender, weightinit, modeltype, traindata, testdata, score , modeliter
    # in that order to the given dictionary.

    #first, make sure the key paths exists
    ensureKeys(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, 'acc'])
    ensureKeys(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, 'acco'])
    ensureKeys(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, '1off'])
    ensureKeys(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, '1offo'])

    #now just enter the keys
    addValue(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, 'acc', modeliter], acc)
    addValue(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, 'acco', modeliter], acco)
    addValue(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, '1off', modeliter], off)
    addValue(resultdict,[ageorgender, weightinit, modeltype, traindata, testdata, '1offo', modeliter], offo)




def addValue(resultdict, keysequence, value):
    ageorgender, weightinit, modeltype, traindata, testdata, scoretype, modeliter = keysequence
    resultdict[ageorgender][weightinit][modeltype][traindata][testdata][scoretype][modeliter] = value
    #print 'key added:', ' -> '.join([str(k) for k in keysequence]) , ':', value



def ensureKeys(resultdict, keysequence):
    ageorgender, weightinit, modeltype, traindata, testdata, scoretype = keysequence

    if not ageorgender in resultdict:
        resultdict[ageorgender] = {}
    if not weightinit in resultdict[ageorgender]:
        resultdict[ageorgender][weightinit] = {}
    if not modeltype in resultdict[ageorgender][weightinit]:
        resultdict[ageorgender][weightinit][modeltype] = {}
    if not traindata in resultdict[ageorgender][weightinit][modeltype]:
        resultdict[ageorgender][weightinit][modeltype][traindata] = {}
    if not testdata in resultdict[ageorgender][weightinit][modeltype][traindata]:
        resultdict[ageorgender][weightinit][modeltype][traindata][testdata] = {}
    if not scoretype in resultdict[ageorgender][weightinit][modeltype][traindata][testdata]:
        resultdict[ageorgender][weightinit][modeltype][traindata][testdata][scoretype] = {}






def build_datastructure():
    results = {}

    #folders = glob.glob('./results-*')
    #for folder in folders:
    txtfiles = glob.glob('./results-*/*.txt')
    T = len(txtfiles)
    for t in xrange(T):
        print 'parsing', t , '/', T, '(', txtfiles[t] ,')'
        readResults(results,txtfiles[t])

    return results

def results2text(results):
    delim = '\t'
    string = ['ageorgender', 'weightinit', 'modeltype', 'trainingdata', 'testdata', 'scoretype', 'modeliter', 'value']
    string = [delim.join(string)]

    print ' formatting data structure to string table ...'
    # [ageorgender][weightinit][modeltype][trainingdata][testdata][score][modeliter]
    aog = sorted(results.keys())
    for a in aog:
        winit = sorted(results[a].keys())
        for w in winit:
            models = sorted(results[a][w].keys())
            for m in models:
                train = sorted(results[a][w][m].keys())
                for tr in train:
                    test = sorted(results[a][w][m][tr].keys())
                    for te in test:
                        scores = sorted(results[a][w][m][tr][te].keys())
                        for s in scores:
                            iters = sorted(results[a][w][m][tr][te][s].keys())
                            for i in iters:
                                string.append(delim.join([a,w,m,tr,te,s,str(i)])  +delim+ str(results[a][w][m][tr][te][s][i]))
    return '\n'.join(string)

def load_results():
    csvpath = './results.csv'
    cachepath = './results.pkl'

    if not os.path.isfile(csvpath):
        results = build_datastructure()
        with open(csvpath,'wb') as f:
            f.write(results2text(results))
    else:
        if not os.path.isfile(cachepath):
            results = {}
            with open(csvpath,'rb') as f:
                lines = f.read().split('\n')[1::]#skip first line. column headers.
                L = len(lines)
                for l in xrange(L):
                    print 'loading', l, '/', L
                    keysnval = lines[l].split()

                    keys = keysnval[:-1]
                    keys[-1] = int(keys[-1])

                    ensureKeys(results,keys[:-1])
                    val = float(keysnval[-1])
                    addValue(results,keys,val)

                with open(cachepath,'wb') as f:
                    #f.write(json.dumps(results))
                    f.write(cPickle.dumps(results))
        else:
            print 'loading results as pickle'
            with open(cachepath,'rb') as f:
                #results = json.loads(f.read())
                results = cPickle.loads(f.read())


    return results


def filterResults(results, ageorgender, weightinit, modeltype, trainingdata, testdata, scoretype, maxiter=None):
    res = results[ageorgender][weightinit][modeltype][trainingdata][testdata][scoretype]
    keys = sorted(res.keys())
    vals = []
    for k in keys:
        vals.append(res[k])

    keys = np.array(keys)
    vals = np.array(vals)


    xbest = np.argmax(vals)
    ybest = vals[xbest]
    xbest = keys[xbest]

    if maxiter is None:
        pass
    else:
        I = np.where(keys <= maxiter)[0]
        keys = keys[I]
        vals = vals[I]


    xshown = keys[-1]
    yshown = vals[-1]
    print 'best results ({}) for [{}][{}][{}][{}][{}]'.format(scoretype,ageorgender,weightinit,modeltype,trainingdata,testdata), ':', 'iter', xbest , 'with', ybest, '(last shown: {}, {}. diff = {}%)'.format(xshown, yshown, (ybest - yshown)*100)
    #print the 20 closest to best models (score wise), ordered by iterations
    difforder = np.argsort(ybest - vals)[:10] #smallest to largest
    bestvals = vals[difforder]
    bestiters = keys[difforder]

    '''
    I = np.argsort(bestiters)[::-1]
    bestvals = bestvals[I]
    bestiters = bestiters[I]
    print [ '({},{})'.format(bestvals[i],bestiters[i]) for i in xrange(len(bestvals))]
    print ''
    '''

    return keys, vals


def plot_finetuning_vs_not(results,problem,train,test,measure,maxIter):
    plt.figure()
    #the baseline from the Gil Levi CVPR2015 paper
    #x,y = filterResults(results,problem,'fromscratch','adiencenet','aligned','aligned',measure,maxiter=maxIter)
    #plt.plot(x,y,label='AdienceNet') #not enoug eval data points


    x,y = filterResults(results,problem,'fromscratch','caffereference',train,test,measure,maxiter=maxIter)
    plt.plot(x,y,color='#1f77b4',alpha=0.5,label='CaffeNet')
    x,y = filterResults(results,problem,'finetuning','caffereference',train,test,measure,maxiter=maxIter)
    plt.plot(x,y,color='#1f77b4',label='CaffeNet FT')

    x,y = filterResults(results,problem,'fromscratch','googlenet',train,test,measure,maxiter=maxIter)
    plt.plot(x,y,color='#ff7f0e', alpha=0.5,label='GoogleNet')
    x,y = filterResults(results,problem,'finetuning','googlenet',train,test,measure,maxiter=maxIter)
    plt.plot(x,y,color='#ff7f0e',label='GoogleNet FT')

    plt.ylabel(measure)
    plt.xlabel('training iterations')

    plt.title('{}, {} vs {}'.format(problem,train,test))

    plt.legend()
    plt.savefig('impactOfFinetuning_{}_{}_vs_{}_{}.pdf'.format(problem,train,test,))




def plot_training_impact(results,problem,model,measure,maxIter,finetuning='finetuning',fromscratch='fromscratch',plot_unaligned=True):
    plt.figure(figsize=(4,4))

    labmap = {  'aligned':'l',\
                'unaligned':'r',\
                'mixed':'m',\
                'fromscratch':'',\
                'finetuning':'n',\
                'imdbwiki':'w'}

    modelmap = {'caffereference':'CaffeNet',\
                'googlenet':'GoogleNet',\
                'vgg16':'VGG-16'}

    #plot model of choice in aligned vs aligned setting
    train = test = 'aligned'
    x,y = filterResults(results,problem,finetuning,model,train,test,measure,maxiter=maxIter)

    ax_labels = plt.gca()

    plotlabel = '[{},{}]'.format(labmap[train],labmap[finetuning])
    plt.plot(x,y,color='#1f77b4',alpha=1.,label=plotlabel)

    if model == 'vgg16':
        plotlabel = '[{},{}]'.format(labmap[train],labmap[fromscratch])
    else:
        plotlabel = '[{}]'.format(labmap[train],labmap[fromscratch])
    x,y = filterResults(results,problem,fromscratch,model,train,test,measure,maxiter=maxIter)
    plt.plot(x,y,color='#1f77b4',alpha=0.5,label=plotlabel)

    if plot_unaligned:
        #plot model of choice in unaligned vs unaligned setting
        train = test = 'unaligned'
        x,y = filterResults(results,problem,finetuning,model,train,test,measure,maxiter=maxIter)
        plotlabel = '[{},{}]'.format(labmap[train],labmap[finetuning])
        plt.plot(x,y,color='#9467bd',alpha=1.,label=plotlabel)

        x,y = filterResults(results,problem,fromscratch,model,train,test,measure,maxiter=maxIter)
        if model == 'vgg16':
            plotlabel = '[{},{}]'.format(labmap[train],labmap[fromscratch])
        else:
            plotlabel = '[{}]'.format(labmap[train],labmap[fromscratch])
        plt.plot(x,y,color='#9467bd',alpha=0.5,label=plotlabel)

    #plot model of choice in mixed vs unaligned setting
    train = 'mixed'
    test = 'unaligned'
    x,y = filterResults(results,problem,finetuning,model,train,test,measure,maxiter=maxIter)
    plotlabel = '[{},{}]'.format(labmap[train],labmap[finetuning])
    plt.plot(x,y,color='#ff7f0e',alpha=1.,label=plotlabel)

    x,y = filterResults(results,problem,fromscratch,model,train,test,measure,maxiter=maxIter)
    if model == 'vgg16':
        plotlabel = '[{},{}]'.format(labmap[train],labmap[fromscratch])
    else:
        plotlabel = '[{}]'.format(labmap[train],labmap[fromscratch])
    plt.plot(x,y,color='#ff7f0e',alpha=0.5,label=plotlabel)








    #SOME PRETTIFICATION AND LAYOUT STUFF
    plt.xlim([0,x[-1]])

    if problem == 'age':
        plt.ylim([0.2,0.7])

        #print baseline results from earliest dnn SOTA to current dnn SOTA
        xb = [0,x[-1]]
        yb = [0.507]*2 #adience DNN paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')

        xb = [0,x[-1]]
        yb = [0.64]*2 #DEX paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')

    else: # problem should be gender
        plt.ylim([0.5,1.0])

        #print baseline results from earliest dnn SOTA to current dnn SOTA
        xb = [0,x[-1]]
        yb = [0.868]*2 #adience DNN paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')

        xb = [0,x[-1]]
        yb = [0.91]*2 #Sighthound paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')



    #plt.ylabel(measure)
    #plt.xlabel('training iterations')


    plt.title('{}: {}'.format(modelmap[model],problem))

    #reorder labels. this, again depends on the models used.
    print ''
    handles, labels =  plt.gca().get_legend_handles_labels()
    print handles,labels
    print ''

    if model == 'caffereference' or model == 'googlenet':
        order = [1, 3, 5, 0, 2, 4]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
        print 'SAY HI'


    plt.legend(handles, labels,loc='lower right')
    #plt.tight_layout()
    #plt.xticks([1,2,3],[1,2,3])

    plt.tight_layout()
    plt.savefig('impactOfPreprocessing_{}_{}_{}.pdf'.format(problem,model,measure))


    def eformat(estr):
         man, ex = estr.replace('+','').split('e')
         ex = ex.lstrip('0')
         return '{}e{}'.format(man,ex)

    #convert xticklabels from written out number to e-notation.
    #1) get tick locations and values
    xticks = [l.get_text() for l in plt.gca().get_xticklabels()]
    xticks = [int(t) for t in xticks]
    xticklabels = [eformat('{:.0e}'.format(t)) for t in xticks]
    plt.xticks(xticks, xticklabels)
    #plt.xticks([1,10000], ['a','b'])
    #ax_labels.set_xticklabels(xticks, xticklabels)

    plt.savefig('impactOfPreprocessing_{}_{}_{}.pdf'.format(problem,model,measure))

    return plt.gca()





def plot_training_impact2(results,problem,model,measure,maxIter):
    plt.figure(figsize=(4,4))

    labmap = {  'aligned':'i',\
                'unaligned':'r',\
                'mixed':'m',\
                'fromscratch':'',\
                'finetuning':'n',\
                'imdbwiki':'w'}

    modelmap = {'caffereference':'CaffeNet',\
                'googlenet':'GoogleNet',\
                'vgg16':'VGG-16'}

    colmap = {  'aligned':'#1f77b4',\
                'unaligned':'#9467bd',\
                'mixed':'#ff7f0e'}



    if model == 'caffereference':
        train = test = 'aligned'

        plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)


        train = test = 'unaligned'

        plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)


        train = 'mixed'

        plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)


    elif model == 'googlenet':

        train = test = 'aligned'

        plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        #x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        #plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle=':')


        train = test = 'unaligned'

        plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=5,label=plotlabel,linewidth=2)


        train = 'mixed'

        plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        #x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        #plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle=':')


    elif model == 'vgg16':

        train = test = 'aligned'

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle='--',linewidth=2)


        train = 'mixed'
        test = 'unaligned'

        plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle='--',linewidth=2)

    else:
        print 'unknown model', model
        exit()



    #SOME PRETTIFICATION AND LAYOUT STUFF
    plt.xlim([0,x[-1]])

    if problem == 'age':
        plt.ylim([0.35,0.65])

        #print baseline results from earliest dnn SOTA to current dnn SOTA
        xb = [0,x[-1]]
        yb = [0.507]*2 #adience DNN paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        plt.text(xb[-1],yb[1],'0.507',fontsize=9,verticalalignment='center')

        xb = [0,x[-1]]
        yb = [0.64]*2 #DEX paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        plt.text(xb[-1],yb[1],'0.64',fontsize=9,verticalalignment='center')

        plt.yticks([.4, .45, .5, .55, .6], ['0.40', '0.45', '0.50', '0.55', '0.60'])

    else: # problem should be gender
        plt.ylim([0.75,0.95])

        #print baseline results from earliest dnn SOTA to current dnn SOTA
        xb = [0,x[-1]]
        yb = [0.868]*2 #adience DNN paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        plt.text(xb[-1],yb[1],'0.868',fontsize=9,verticalalignment='center')

        xb = [0,x[-1]]
        yb = [0.91]*2 #Sighthound paper baseline
        plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        plt.text(xb[-1],yb[1],'0.91',fontsize=9,verticalalignment='center')

        plt.yticks([.75, .8, .85, .9, .95], ['0.75', '0.80', '0.85', '0.90', '0.95'])


    #plt.ylabel(measure)
    #plt.xlabel('training iterations')


    plt.title('{}: {}'.format(modelmap[model],problem))

    #reorder labels. this, again depends on the models used.
    print ''
    handles, labels =  plt.gca().get_legend_handles_labels()
    print handles,labels
    print ''

    #if model == 'caffereference' or model == 'googlenet':
    #    order = [1, 3, 5, 0, 2, 4]
    #    handles = [handles[i] for i in order]
    #    labels = [labels[i] for i in order]


    plt.legend(handles, labels,loc='lower right')
    #plt.tight_layout()
    #plt.xticks([1,2,3],[1,2,3])

    plt.tight_layout()
    plt.savefig('impactOfPreprocessing_{}_{}_{}.pdf'.format(problem,model,measure))


    def eformat(estr):
         man, ex = estr.replace('+','').split('e')
         ex = ex.lstrip('0')
         return '{}e{}'.format(man,ex)

    #convert xticklabels from written out number to e-notation.
    #1) get tick locations and values
    xticks = [l.get_text() for l in plt.gca().get_xticklabels()]
    xticks = [int(t) for t in xticks]
    xticklabels = [eformat('{:.0e}'.format(t)) for t in xticks]
    plt.xticks(xticks, xticklabels)
    #plt.xticks([1,10000], ['a','b'])
    #ax_labels.set_xticklabels(xticks, xticklabels)

    plt.savefig('impactOfPreprocessing_{}_{}_{}.pdf'.format(problem,model,measure))

    return plt.gca()








def plot_training_impact_bars(results,problem,model,measure,maxIter=10^9):
    plt.figure(figsize=(4,4))

    labmap = {  'aligned':'i',\
                'unaligned':'r',\
                'mixed':'m',\
                'fromscratch':'',\
                'finetuning':'n',\
                'imdbwiki':'w'}

    modelmap = {'caffereference':'CaffeNet',\
                'googlenet':'GoogleNet',\
                'vgg16':'VGG-16'}

    colmap = {  'aligned':'#1f77b4',\
                'unaligned':'#9467bd',\
                'mixed':'#ff7f0e'}



    if model == 'caffereference':
        train = test = 'aligned'

        #plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        y_a = y.max()
        x,y = filterResults(results,problem,'fromscratch',model,train,test,'1offo',maxiter=maxIter)
        y_a1 = y.max()
        c_a = colmap[train] + '80' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_af = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_af1 = y.max()
        c_af = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)


        train = test = 'unaligned'

        #plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        y_u = y.max()
        x,y = filterResults(results,problem,'fromscratch',model,train,test,'1offo',maxiter=maxIter)
        y_u1 = y.max()
        c_u = colmap[train] + '80' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_uf = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_uf1 = y.max()
        c_uf = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)


        train = 'mixed'

        #plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        y_m = y.max()
        x,y = filterResults(results,problem,'fromscratch',model,train,test,'1offo',maxiter=maxIter)
        y_m1 = y.max()
        c_m = colmap[train] + '80' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_mf = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_mf1 = y.max()
        c_mf = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        y = [y_a, y_u, y_m,  y_af, y_uf, y_mf]
        y1 = [y_a1, y_u1, y_m1,  y_af1, y_uf1, y_mf1]
        colors = [c_a, c_u, c_m,  c_af, c_uf, c_mf]
        x = [1,2,3, 5,6,7]


    elif model == 'googlenet':

        train = test = 'aligned'

        #plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        #i = np.argmax(y)
        #x_a = x[i]
        #y_a = y[i]
        y_a = y.max()
        x,y = filterResults(results,problem,'fromscratch',model,train,test,'1offo',maxiter=maxIter)
        y_a1 = y.max()
        c_a = colmap[train] + '80' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_af = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_af1 = y.max()
        c_af = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        #x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        #plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle=':')




        train = test = 'unaligned'

        #plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        y_u = y.max()
        x,y = filterResults(results,problem,'fromscratch',model,train,test,'1offo',maxiter=maxIter)
        y_u1 = y.max()
        c_u = colmap[train] + '80' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_uf = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_uf1 = y.max()
        c_uf = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=5,label=plotlabel,linewidth=2)


        train = 'mixed'

        #plotlabel = '[{}]'.format(labmap[train],labmap['fromscratch'])
        x,y = filterResults(results,problem,'fromscratch',model,train,test,measure,maxiter=maxIter)
        y_m = y.max()
        x,y = filterResults(results,problem,'fromscratch',model,train,test,'1offo',maxiter=maxIter)
        y_m1 = y.max()
        c_m = colmap[train] + '80' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=.5,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_mf = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_mf1 = y.max()
        c_mf = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        #x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        #plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle=':')

        y = [y_a, y_u, y_m,  y_af, y_uf, y_mf]
        y1 = [y_a1, y_u1, y_m1,  y_af1, y_uf1, y_mf1]

        colors = [c_a, c_u, c_m,  c_af, c_uf, c_mf]
        x = [1,2,3, 5,6,7]
        #plt.bar(x, y, align = 'center', color=colors, edgecolor='#00000000')


    elif model == 'vgg16':

        train = test = 'aligned'

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_uf = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_uf1 = y.max()
        c_uf = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        y_ui = y.max()
        x,y = filterResults(results,problem,'imdbwiki',model,train,test,'1offo',maxiter=maxIter)
        y_ui1 = y.max()
        c_ui = colmap[train] + 'c8' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle='--',linewidth=2)


        train = 'mixed'
        test = 'unaligned'

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['finetuning'])
        x,y = filterResults(results,problem,'finetuning',model,train,test,measure,maxiter=maxIter)
        y_mf = y.max()
        x,y = filterResults(results,problem,'finetuning',model,train,test,'1offo',maxiter=maxIter)
        y_mf1 = y.max()
        c_mf = colmap[train] + 'ff' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1,label=plotlabel,linewidth=2)

        #plotlabel = '[{},{}]'.format(labmap[train],labmap['imdbwiki'])
        x,y = filterResults(results,problem,'imdbwiki',model,train,test,measure,maxiter=maxIter)
        y_mi = y.max()
        x,y = filterResults(results,problem,'imdbwiki',model,train,test,'1offo',maxiter=maxIter)
        y_mi1 = y.max()
        c_mi = colmap[train] + 'c8' # add alpha
        #plt.plot(x,y,color=colmap[train],alpha=1.,label=plotlabel, linestyle='--',linewidth=2)

        y = [y_uf, y_mf,  y_ui, y_mi]
        y1 = [y_uf1, y_mf1,  y_ui1, y_mi1]
        colors = [c_uf, c_mf,  c_ui, c_mi]

        x = [1,2, 6,7]

    else:
        print 'unknown model', model
        exit()




    #SOME PRETTIFICATION AND LAYOUT STUFF
    #plt.xlim([0,x[-1]])

    if problem == 'age':
        plt.ylim([0.46,0.65])

        #print baseline results from earliest dnn SOTA to current dnn SOTA
        xb1 = [min(x) -1]
        yb1 = [0.507] #adience DNN paper baseline
        yb11 = [0.847]

        #plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        #plt.text(xb[-1],yb[1],'0.507',fontsize=9,verticalalignment='center')

        xb2 = [max(x)+1]
        yb2 = [0.64] #DEX paper baseline
        yb21 = [0.966]
        #plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        #plt.text(xb[-1],yb[1],'0.64',fontsize=9,verticalalignment='center')

        #plt.yticks([.4, .45, .5, .55, .6], ['0.40', '0.45', '0.50', '0.55', '0.60'])

        x = xb1 + x + xb2
        y = yb1 + y + yb2
        y1 = yb11 + y1 + yb21
        c = ['#80808080'] + colors + ['#80808080']
        plt.bar(x, y,  align = 'center', color=c, edgecolor='#00000000')

        for i in xrange(len(x)):
            plt.text(x[i],y[i],'{:.1f}'.format(y[i]*100),fontsize=8,verticalalignment='bottom',horizontalalignment='center', color=c[i])
            plt.text(x[i],y[i]-0.001,'({:.1f})'.format(y1[i]*100),fontsize=6,verticalalignment='top',horizontalalignment='center', color='#ffffffff')


        plt.yticks([.48, .52, .56, .60, .64], ['0.48', '0.52', '0.56', '0.60', '0.64'])



    else: # problem should be gender
        plt.ylim([0.85,0.95])

        #print baseline results from earliest dnn SOTA to current dnn SOTA
        xb1 = [min(x) -1]
        yb1 = [0.868] #adience DNN paper baseline
        #plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        #plt.text(xb[-1],yb[1],'0.868',fontsize=9,verticalalignment='center')

        xb2 = [max(x)+1]
        yb2 = [0.91] #Sighthound paper baseline
        #plt.plot(xb,yb,color='#000000',alpha=1,linewidth = 0.5,linestyle='--')
        #plt.text(xb[-1],yb[1],'0.91',fontsize=9,verticalalignment='center')

        x = xb1 + x + xb2
        y = yb1 + y + yb2
        c = ['#80808080'] + colors + ['#80808080']

        #plt.bar(x, y, align = 'center', color=colors, edgecolor='#00000000')
        plt.bar(x, y,  align = 'center', color=c, edgecolor='#00000000')

        for i in xrange(len(x)):
            plt.text(x[i],y[i],'{:.1f}'.format(y[i]*100),fontsize=8,verticalalignment='bottom',horizontalalignment='center', color=c[i])

        #plt.yticks([.75, .8, .85, .9, .95], ['0.75', '0.80', '0.85', '0.90', '0.95'])
    plt.xticks([],[])


    #plt.ylabel(measure)
    #plt.xlabel('training iterations')


    plt.title('{}: {}'.format(modelmap[model],problem))

    #reorder labels. this, again depends on the models used.
    print ''
    handles, labels =  plt.gca().get_legend_handles_labels()
    print handles,labels
    print ''

    #if model == 'caffereference' or model == 'googlenet':
    #    order = [1, 3, 5, 0, 2, 4]
    #    handles = [handles[i] for i in order]
    #    labels = [labels[i] for i in order]


    #plt.legend(handles, labels,loc='lower right')
    #plt.tight_layout()
    #plt.xticks([1,2,3],[1,2,3])

    #plt.tight_layout()
    #plt.savefig('impactOfPreprocessing_{}_{}_{}.pdf'.format(problem,model,measure))


    def eformat(estr):
         man, ex = estr.replace('+','').split('e')
         ex = ex.lstrip('0')
         return '{}e{}'.format(man,ex)

    #convert xticklabels from written out number to e-notation.
    #1) get tick locations and values
    #xticks = [l.get_text() for l in plt.gca().get_xticklabels()]
    #xticks = [int(t) for t in xticks]
    #xticklabels = [eformat('{:.0e}'.format(t)) for t in xticks]
    #plt.xticks(xticks, xticklabels)
    #plt.xticks([1,10000], ['a','b'])
    #ax_labels.set_xticklabels(xticks, xticklabels)



    plt.savefig('impactOfPreprocessing_{}_{}_{}_BARPLOT.pdf'.format(problem,model,measure))

    return plt.gca()







''' MAIN '''
#scan_files() # files seem to be ok
results = load_results()


#[ageorgender][weightinit][modeltype][trainingdata][testdata][scoretype][modeliter]
#only use keys until the score bit. we need probably all model iterations.
maxIter = 1e5
measure = 'acco'


model = 'googlenet'
#ax_ga = plot_training_impact2(results,'age', model, measure,maxIter)
#ax_gg = plot_training_impact2(results,'gender', model,  measure,maxIter)
ax_gg = plot_training_impact_bars(results,'gender', model,  measure,maxIter)
ax_ga = plot_training_impact_bars(results,'age', model,  measure,maxIter)


model = 'vgg16'
#ax_va = plot_training_impact2(results,'age', model, measure,maxIter)
#ax_vg = plot_training_impact2(results,'gender', model, measure,maxIter)
ax_gg = plot_training_impact_bars(results,'gender', model,  measure,maxIter)
ax_ga = plot_training_impact_bars(results,'age', model,  measure,maxIter)

model = 'caffereference'

#ax_ca = plot_training_impact2(results,'age', model, measure,maxIter)
#ax_cg = plot_training_impact2(results,'gender', model, measure,maxIter)
ax_gg = plot_training_impact_bars(results,'gender', model,  measure,maxIter)
ax_ga = plot_training_impact_bars(results,'age', model,  measure,maxIter)


