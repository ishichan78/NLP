#NLP

##目的・できること
* ###研究室みんなで機能増やしながら理解を深める

現時点ではテキストファイルからSHOGUNなどに適応可能なデータを作成できるようにしています。 
使い勝手が良ければ、研究室で共有できるDropboxのフォルダ作ってその中に保存して、`$ ln -s path/to/share/dir/okdlab.py /usr/local/lib/python2.7/dist-packages/`みたいな感じにして使えばいいと思うよ 
使い方についてはpydocやインタプリタ上のhelpを見てください 
下にも少し説明を書いときます 

##必要なもの
* Python &gt;= 2.7
* NumPy
* MeCab

## Description and Usage

###SaveLoad
* わりとおなじみの 
* objectクラスにsave,load機能を与えたクラス
* 作成するたいていのクラスにはこれを継承させる
* 保存はpickle使ってます

###ZipType
* なんでもないクラス
* 圧縮ファイル対応させるために作っただけ

###Text
* gensimに与えるようなデータを作成できる
* gensimなくてもTFIDFくらいまでならこのライブラリでなんとかなりそう

        text = Text(fname)
        or
        text = Text.load_file(fname)

とりあえずデフォルトでは１行１文書形式で記述されたテキストファイルから作成するようにしてあります。ファイルの処理を変更したい場合はload\_fileにprocess引数としてそのメソッドを与えます。 
zip,gz,bz2ファイルを読み込む場合は、

    text = Text(gz_fname, is_compressed=True, zip_type="gz")
    or
    text = Text.load_file(fname, is_compressed=True, zip_type="gz")

みたいな感じで
英語の同様のファイルを読み込みたいなら

    def en_load(f):
        return [line.rstrip('\n').split() for line in f if line]
    text = Text(fname, process=en_load)

みたいにしたらできそうな気がします 
lambda式使っても大丈夫かな? 

###Corpus
* SHOGUNに使用できるデータが作成できると思う
* TextやText.load\_fileのやつを受け取れる
* 辞書はvocabを指定することで特定のものを使用できる
* 出現回数・頻度・TFIDFに対応
* SVMLightの入力形式にファイル出力できる

        corpus = Corpus(Text(fname))
        or
        text = Text.load_file(fname)
        corpus = Corpus(text)

あとは**corpus.dense**に２次元配列ができてて、別に用意したラベルと一緒にSHOGUNに渡せば学習できると思う
