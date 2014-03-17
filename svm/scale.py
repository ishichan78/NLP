#!/usr/bin/env python
#-*- coding:utf-8 -*-

from numpy import array,ones,concatenate,mean,log2 as log
try: import cPickle as pickle
except ImportError: import pickle

class Scale:
    def __init__(self,outs,labels):
        self.outs = outs
        self.labels = labels
        self.pre = self.precision(outs,labels)
        self.rec = self.recall(outs,labels)
        self.acc = self.accuracy(outs,labels)

    @staticmethod
    def extract(outs,labels):
        tp,fp,tn,fn = 0,0,0,0
        for out,label in zip(outs,labels):
            if int(out) == 1:
                if int(label) == int(out):
                    tp += 1
                else:
                    fp += 1
            else:
                if int(label) == int(out):
                    tn += 1
                else:
                    fn += 1
        return tp,fp,tn,fn

    @classmethod
    def precision(cls,outs,labels,lab=0):
        tp,fp,tn,fn = cls.extract(outs,labels)
        p = 1.0*tp/(tp+fp) if tp+fp else 0
        n = 1.0*tn/(tn+fn) if tn+fn else 0
        if not lab:
            return p,n
        elif lab == 1:
            return p
        elif lab == -1:
            return n
    
    @classmethod
    def recall(cls,outs,labels,lab=0):
        tp,fp,tn,fn = cls.extract(outs,labels)
        t = 1.0*tp/(tp+fn) if tp+fn else 0
        f = 1.0*tn/(tn+fp) if tn+fp else 0
        if not lab:
            return t,f
        elif lab == 1:
            return t
        elif lab == -1:
            return f

    @classmethod
    def accuracy(cls,outs,labels):
        tp,fp,tn,fn = cls.extract(outs,labels)
        return 1.0*(tp+tn)/(tp+fp+tn+fn)


class Distance:
    @staticmethod
    def kl_divergence(X,P,Q):
        sum = 0
        for x in X:
            sum += 1.0*P[x]*log(1.0*P[x]/Q[x])
        return sum

    @staticmethod
    def js_divergence(X,P,Q):
        sum1 = 0
        sum2 = 0
        for x in X:
            sum1 += 1.0*P[x]*log(P[x]/(0.5*(P[x]+Q[x])))
            sum2 += 1.0*Q[x]*log(Q[x]/(0.5*(P[x]+Q[x])))
        return 0.5 * (sum1+sum2)

def load_pickle(fname):
    with file(fname, 'rb') as f:
        return pickle.load(f)

if __name__=='__main__':
    import sys,os

    dirname = os.getcwd()
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        dirname = sys.argv[1]
    fnames = map(lambda fname: os.path.join(dirname,fname), os.listdir(dirname))
    
    outs = [load_pickle(fname) for fname in fnames if ".pickle" in fname]
    labels = concatenate((ones(len(outs[0])/2),-ones(len(outs[0])/2)))
    scales = [Scale(out,labels) for out in outs]
    precisions,recalls = {},{}
    precisions['pos'],precisions['neg'] = zip(*map(lambda scale:scale.pre,scales))
    recalls['pos'],recalls['neg']       = zip(*map(lambda scale:scale.rec,scales))
    accuracies = map(lambda scale:scale.acc,scales)

    print "precision"
    for pos,neg in zip(precisions['pos'],precisions['neg']):
        print "%0f\t%0f"%(pos,neg)
    print "mean: %0f\t%0f\t%0f\n"%(
                mean(precisions['pos']),
                mean(precisions['neg']),
                mean(precisions['pos']+precisions['neg']))

    print "recall"
    for pos,neg in zip(recalls['pos'],recalls['neg']):
        print "%0f\t%0f"%(pos,neg)
    print "mean: %0f\t%0f\t%0f\n"%(
                mean(recalls['pos']),
                mean(recalls['neg']),
                mean(recalls['pos']+recalls['neg']))
    
    print "accuaracy"
    for accuracy in accuracies:
        print "%0f"%accuracy
    print "mean: %0f"%mean(accuracies)
