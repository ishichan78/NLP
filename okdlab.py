#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import re
import zipfile, gzip, bz2
try: import cPickle as pickle
except ImportError: import pickle

import numpy as np
import MeCab

### SaveLoad
class SaveLoad(object):
    """
    オブジェクトクラスにsave,loadを付け加える
    """
    def save(self, fname):
        with file(fname, 'wb') as f:
            pickle.dump(self, f, protocol=-1)

    @staticmethod
    def load(fname):
        with file(fname, 'rb') as f:
            return pickle.load(f)

### ZipType
class ZipType:
    """
    圧縮ファイルを開くメソッドを返すだけのクラス
    """
    @staticmethod
    def zip():
        return zipfile.ZipFile

    @staticmethod
    def gz():
        return gzip.GzipFile

    @staticmethod
    def bz2():
        return bz2.BZ2File

### tokenize method
def jatokenize(text, pos={"名詞","動詞","形容詞"}, stop_words={}):
    tagger = MeCab.Tagger()
    for sentence in text:
        if not sentence: continue

        node = tagger.parseToNode(sentence).next
        while node:
            if node.feature.split(',', 1)[0] in pos:
                try:
                    surface = node.surface.lower()
                    if len(surface) and not surface in stop_words:
                        yield surface
                except:
                    pass
            node = node.next

def tokenize(text, tokenizer=jatokenize, kw={}):
    return [token for token in tokenizer(text, **kw)
                if not token.startswith('_')]

### Text
class Text(SaveLoad):
    """
    docs: 
    [[w_11,w_12,...,w_1n_1],
     [w_21,...,w_2n_2],...,
     [w_d1,...,w_dn_d]]
    なデータモデル
    """
    def __init__(self, fname, **args):
        self.docs = self.load_file(fname, **args)

    def __getitem__(self, key):
        return self.docs[key]

    def __len__(self):
        return len(self.docs)

    def __str__(self):
        s = "["
        for doc in self.docs:
            s += "[{doc}]\n".format(doc=','.join(doc))
        s += "]"
        return s

    @classmethod
    def load_file(cls, fname,
                  is_compressed=False, zip_type=None,
                  process=None, args={}):
        """
        ファイル名とファイル形式を指定してテキストコーパスを
        作成するメソッド

        fname: ファイル名
        is_compressed: 圧縮ファイルかどうか
        zip_type: 圧縮形式(zip, gz, bz2)
        process: ファイルからコーパスを作成するメソッド
        args: processに渡す引数

        """
        file = getattr(ZipType, zip_type)() \
                if is_compressed else open
        try:
            f = file(fname)
            process = process if process else cls._load_sample
            text = process(f, args)
        
        finally:
            if f:
                f.close()

        return text

    @staticmethod
    def _load_sample(f, args):
        """
        １行１文書形式で書かれたファイルからコーパス(Text.docs)を
        作成するメソッド
        """
        if not args:
            args = {'pos': {"名詞","動詞","形容詞"}}
        return [tokenize(doc, kw=args) for doc in (
            line.rstrip('\n').replace('<br/>','\n').split('\n')
                for line in f)]

### Corpus
class Corpus(SaveLoad):
    """
    Textや同様のデータからshogunなどが利用できる
    コーパスを作成することができるやつ
    D: 文書数
    V: 語彙数
    dtype: データタイプ（bow,tf,tfidf）
        bow:   出現回数
        tf:    頻度
        tfidf: TF-IDF (log_2(|D|/(tが含まれる文書数)))
        割ったりlogの計算があるので変換後は元に戻せません
    docs: Text
    vocab: 語彙
      複数コーパスで辞書を統一したい場合はこれを指定する
    token2id: 語彙
    dense: SHOGUNとかで使えるデータ(numpy.ndarray)

    **
    dump_svmlightはオブジェクトの特徴をファイルに出力するものですが
    出力したファイルからオブジェクトを復元できるわけではありません
    復元することを想定している場合は save, load を使ってください
    **
    """
    def __init__(self, text, vocab=None, dtype="bow"):
        self.docs = text
        self.dtype = "bow"
        self.vocab = vocab if vocab else tuple(set(sum(text, [])))
        self.token2id = dict([(v,i) for i,v in enumerate(self.vocab)])
        self.D = len(text)
        self.V = len(self.vocab)
        self.dense = np.zeros((self.V, self.D), dtype=np.float64)
        
        for j,doc in enumerate(text):
            for token in doc:
                self.dense[self.token2id[token], j] += 1.0

        if dtype == "tfidf":
            self.tfidf()
        elif dtype == "tf":
            self.tf()

    def __getitem__(self, key):
        return self.dense[key]

    def __len__(self):
        return len(self.dense)

    def __str__(self):
        return "dtype: {dtype}\n{dense}".format(
                                dtype=self.dtype, dense=self.dense)

    def tf(self):
        if self.dtype == "tfidf":
            sys.stderr.write("dtype is tfidf. can't convert.\n")
            return

        elif self.dtype == "bow":
            for d in xrange(self.D):
                N = self.dense[:,d].sum()
                self.dense[:,d] = self.dense[:,d]/N if N else 0.0

        self.dtype = "tf"
        return self.dense

    def tfidf(self):
        """
        denseの素性値を単語出現回数からtfidf値に変える
        """
        if self.dtype == "tfidf":
            return self.dense

        ## tf
        elif self.dtype == "bow":
            for d in xrange(self.D):
                N = self.dense[:,d].sum()
                self.dense[:,d] = self.dense[:,d]/N if N else 0.0
        
        ## tfidf
        for t in xrange(self.V):
            df = (self.dense[t,:]>0.0).sum()
            idf = np.log2(1.0*self.D/df) if df else 0.0
            self.dense[t,:] *= idf
        
        self.dtype = "tfidf"
        return self.dense

    def dump_svmlight(self, pref, lab=None):
        """
        SVMLightの入力形式にファイル出力します
        出力ファイル：.svmlight, .dic
        １行が "label token_id:score token_id:score ..." の形式
        """
        if not lab:
            lab = [1 for i in xrange(self.D)]
        if len(lab) != self.D:
            sys.stderr.write("invalid num of labels\n")
            return
        
        with file(pref+'.svmlight', 'w') as f:
            for j in xrange(self.D):
                s = str(lab[j])
                for i in self[:,j].nonzero()[0]:
                    s += " {i}:{score}".format(i=i, score=self[i,j])
                f.write(s+'\n')

        with file(pref+'.dic', 'w') as f:
            for token in self.vocab:
                f.write("{token}\n".format(token=token))

    @classmethod
    def load_svmlight(cls, pref, pref_dic=None):
        """
        **
        dump_svmlightで作成したファイルを読み込むメソッド
        Corpusのインスタンスとして復元できるわけではないので注意
        **

        dense: Corpus.dense
        vocab: Corpus.vocab
        lab: ラベル
        """
        if not pref_dic: pref_dic = pref
        vocab = cls.load_dictionary(pref_dic)
        V = len(vocab)
        with file(pref+'.svmlight') as f:
            lab = [int(line.split(' ',1)[0]) for line in f if line]
            D = len(lab)
        dense = np.zeros((V,D), dtype=np.float64)
        p = re.compile(r"([0-9]+):([0-9\.]+)")
        print V,D
        with file(pref+'.svmlight') as f:
            for j,line in enumerate(f):
                for (i,score) in p.findall(line):
                    dense[int(i),j] = np.float64(score)
        return dense, vocab, lab

    @staticmethod
    def load_dictionary(pref):
        with file(pref+'.dic') as f:
            return [line.rstrip('\n') for line in f if line]
