#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import os
import re

import MeCab
import gensim

def jatokenize(text,pos):
    tagger = MeCab.Tagger()
    for sentence in text:
        node = tagger.parseToNode(sentence).next
        while node:
            if node.feature.split(',',1)[0] in pos:
                try:
                    surface = node.surface.lower()
                    if len(surface): yield surface
                except:
                    pass
            node = node.next

def tokenize(content,pos):
    return [token for token in jatokenize(content,pos) if not token.startswith('_')]


def gen_documents(f):
    return (line.rstrip('\n').replace('<br/>','\n').split('\n') for line in f)

def mk_texts(f,pos):
    return [[word for word in tokenize(document,pos)] for document in gen_documents(f)]

def rm_ncnt_tokens(texts,n=1):
    all_tokens = sum(texts,[])
    tokens_nth = set(word for word in set(all_tokens) if all_tokens.count(word) == n)
    texts = [[word for word in text if word not in tokens_nth] for text in texts]
    return texts
    
def mk_dict(texts):
    dictionary = gensim.corpora.Dictionary(texts)
    return dictionary

def mk_corpus(texts,dictionary,is_tfidf=False):
    corpus = [doc2bow for doc2bow in gen_corpus(texts,dictionary)]
    if is_tfidf:
        tfidf = gensim.models.TfidfModel(corpus)
        corpus = tfidf[corpus]
    return corpus

def gen_corpus(texts,dictionary):
    for text in texts:
        yield dictionary.doc2bow(text)

def save_i2t(tokens,f):
    for i,token in enumerate(tokens):
        f.write(u"{id} {token}\r\n".format(id=i,token=token).encode('utf-8'))

def load_i2t(f):
    dic = {}
    for line in f:
        i2t = line.decode('utf-8').rstrip('\n').split(' ')
        if len(i2t) is 2:
            dic[i2t[1]] = int(i2t[0])
    return dic


def mk_matrix(documents,dictionary,pos):
    sys.stderr.write("matrix: {0} {1}\n".format(len(documents),len(dictionary.items())))
    row,col = len(documents),len(dictionary)
    matrix = numpy.zeros((row,col),dtype=int)
    for i,document in enumerate(documents):
        for word in tokenize(document,pos):
            try:
                matrix[i][dictionary[word]] += 1
            except Exception as e:
                continue
    return matrix

def save_corpus(corpus, pref, num=0):
    corpus_fname = pref+'.svmlight'
    gensim.corpora.SvmLightCorpus.serialize(corpus_fname, corpus)
    with file(corpus_fname,'rU') as f_in, file(corpus_fname+'.tmp','w') as f_out:
        for i,line in enumerate(f_in):
            lst = line.split(' ')
            if i < num:
                lst[0] = "1"
            else:
                lst[0] = "-1"
            line = ' '.join(lst)
            sys.stderr.write("{0}\n".format(line))
            f_out.write(line)
    os.rename(corpus_fname+'.tmp',corpus_fname)

def load_corpus(fname):
    return gensim.corpora.SvmLightCorpus(fname)

if __name__=='__main__':
    from optparse import OptionParser,OptionValueError
    import numpy

    usage = "usage: %prog [options] keyword"
    parser = OptionParser(usage)

    parser.add_option("--f1",dest="fname1",action="store",type="string",help="documents' file name")
    parser.add_option("--f2",dest="fname2",action="store",type="string",help="documents' file name")
    parser.add_option("-p","--pos", dest="pos", default='名詞,動詞,形容詞',action="store",type="string",help="part of speach")
    parser.add_option("-o","--pref",dest="pref",default='out', action="store",type="string",help="output file's prefix")

    parser.add_option('-b', "--bow",  dest="tfidf", action="store_false", default=False)
    parser.add_option('-t', "--tfidf",dest="tfidf", action="store_true",  default=False)

    (opts,args) = parser.parse_args()
    opts.pos = opts.pos.split(',')

    with file(opts.fname1,'rU') as f1, file(opts.fname2,'rU') as f2:
        num_pos = f1.read().count('\n')
        num_neg = f2.read().count('\n')
        print "pos:",num_pos
        print "neg:",num_neg
        f1.seek(0);f2.seek(0)
        texts = mk_texts(f1,opts.pos)+mk_texts(f2,opts.pos)

    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.save(opts.pref+'.dic')

    corpus = mk_corpus(texts,dictionary,opts.tfidf)
    save_corpus(corpus, opts.pref, num_pos)
    sys.stderr.write("all done\n")
