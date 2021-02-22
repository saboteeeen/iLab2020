#MeCabをインストール
import MeCab

#形態素解析したい文章
data = "すもももももももものうち"

#MeCabを呼び出す(Taggerの引数は辞書の指定)
mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd').parse(data)
#改行ごとに文章を分割
lines = mecab.split('\n')
#各行ごとに文章の構成単位に分解
items = (re.split('[\t]',line) for line in lines)
#形態素解析した結果を表示
for item in items:
    print(item)