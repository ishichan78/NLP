#!/usr/bin/env python
#-*- coding:utf-8 -*-
# shogun = 2.1.0

import numpy
import gensim
from shogun.Features import RealFeatures, BinaryLabels
from shogun.Kernel import GaussianKernel, LinearKernel, PolyKernel, ExponentialKernel, SigmoidKernel
from shogun.Distance import EuclideanDistance
from shogun.Classifier import LibSVM

def load_lab(fname):
    with file(fname) as f:
        return [int(line.split(' ',1)[0]) for line in f]

def load_bnst(fname):

"""
def devide(data,n):
    dev = []
    for i in range(n): dev.append([])
    data = numpy.array(data).T
    for i in range(len(data)):
        k = i%n
        dev[k].append(data[i])
    dev = map(lambda x:numpy.array(x,dtype=numpy.float64).T, dev)
    return dev
"""

def devide(data,n,k):
    train,test = [],[]
    data = numpy.array(data).T
    for i in range(len(data)):
        if i % n - k:
            train.append(data[i])
        else:
            test.append(data[i])
    train = numpy.array(train,dtype=numpy.float64).T
    test  = numpy.array(test, dtype=numpy.float64).T
    return train,test

## kernel
class Kernel(object):
    @staticmethod    
    def gauss(feats, width=2.1):
        return GaussianKernel(feats, feats, width)
    
    @staticmethod
    def linear(feats):
        return LinearKernel(feats,feats)

    @staticmethod
    def exponential(feats, tau_coef=1.0, width=10):
        distance = EuclideanDistance(feats, feats)
        return ExponentialKernel(feats, feats, tau_coef, distance, width)

    @staticmethod
    def poly(feats, degree=4, inhomogene=False, use_normalization=True):
        return PolyKernel(feats, feats, degree, inhomogene, use_normalization)

    @staticmethod
    def sigmoid(feats, size=10, gamma=1.2, coef0=1.3):
        return SigmoidKernel(feats, feats, size, gamma, coef0)


## SVM
def svm_train(kernel, lab, C=1):
    labels = BinaryLabels(lab)
    svm = LibSVM(C,kernel,labels)
    svm.train()
    return svm

def svm_test(svm,feats_train,feats_test):
    kernel = svm.get_kernel()
    kernel.init(feats_train,feats_test)
    return svm.apply(feats_test).get_labels()

## CROSS VALIDATION
def cross_validation(data,lab,n,args={}):
    for i in range(n):
        sys.stderr.write("create data\n")
        train_dat,test_dat = devide(data,n,i)
        train_lab,test_lab = devide(lab,n,i)
        train_feats,test_feats = RealFeatures(train_dat),RealFeatures(test_dat)

#        kernel = gauss(train_feats,args.get('width',2.1))
        kernel = getattr(Kernel, args['kernel'])(train_feats)
        svm = svm_train(kernel,train_lab,args.get('C',1))
        out = svm_test(svm,train_feats,test_feats)
        args.get('callback',do_nothing)(out,test_lab,{'i':i, 'kernel':args['kernel']})

def do_nothing(dummy):
    pass

def save_out(out,lab,args):
    fname = "{name}_{i:d}.pickle".format(name=args['kernel'], i=args['i'])
    sys.stderr.write(fname+"pickling\n")
    with file(fname,'wb') as f:
        pickle.dump(out,f,protocol=-1)

## MAIN
if __name__=='__main__':
    import sys
    from optparse import OptionParser,OptionValueError
    try: import cPickle as pickle
    except ImportError: import pickle

    usage = "usage: %prog [options] keyword"
    parser = OptionParser(usage)

    parser.add_option("-n",dest="n",action="store",type="int",default=8,help="交差検定をするときの分割数")
    parser.add_option("-i","--pref",action="store",default="out",type="string",help="file name prefix")
    parser.add_option("-k","--kernel",action="store",type="string",default="gauss",help="交差検定をするときの分割数")

    (opts,args) = parser.parse_args()

    print args
    input()

    dictionary = gensim.corpora.Dictionary.load(opts.pref+".dic")
    corpus = gensim.corpora.SvmLightCorpus(opts.pref+".svmlight")
    lab = load_lab(opts.pref+".svmlight")
    num_terms = len(dictionary)
    num_texts = len(corpus)
    matrix = gensim.matutils.corpus2dense(corpus,num_terms)
#hstack(matrix, bn)
    cross_validation(matrix,lab,opts.n,{'callback':save_out, 'kernel':opts.kernel})

    """
    features = RealFeatures(matrix)
    labels = BinaryLabels(lab)
    kernel = GaussianKernel()
    classifier = LibSVM()

    splitting_strategy = StratifiedCrossValidationSplitting(labels, opts.n)
    evaluation_criterium = ContingencyTableEvaluation(ACCURACY)
    cross_validation = CrossValidation(classifier, features, labels, splitting_strategy, evaluation_criterium)
    cross_validation.set_autolock(False)

    # cross_validation.set_num_runs(10)
    # cross_validation.set_conf_int_alpha(0.05)

    result = cross_validation.evaluate()
    print "mean:",result.mean
    if result.has_conf_int:
        print "[",result.conf_int_low,",",result.conf_int_up,"] with alpha=", result.conf_int_alpha
    """
