import sys
sys.path.append('..')
from common import config
config.GPU = False
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from simple_Lstmlm import SimpleLstmlm
from common.np import *  # import numpy as np (or import cupy as np)
from tangoatume import preprocess
from common.util import eval_perplexity, to_gpu

import time
#import pandas as pd
#import openpyxl



mode = int(input("収束条件有りなら1、学習回数決め打ちは0:"))
#ハイパーパラメータの設定
batch_size = 20
wordvec_size = 650
hidden_size = 650 #RNNの隠れ状態ベクトルの要素数
time_size = 35 #RNNを展開するサイズ
lr = 20.0
max_epoch = 1
max_grad = 0.25



#学習データの読み込み
file_name = "./text/sample.txt"
file = open(file_name,encoding='shift-jis')
corpus, word_to_id, id_to_word = preprocess(file)

if config.GPU:
    corpus = to_gpu(corpus)

vocab_size = len(word_to_id)
corpus_size = len(corpus)

print('corpus_size: %d, vocabulary size: %d' % (corpus_size, vocab_size))
#print(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

#モデルの生成
model = SimpleLstmlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer) #モデルの学習

#勾配クリッピングを適用して学習
best_ppl = float('inf')
t1 = time.time()

for epoch in range(max_epoch):
    flag = trainer.fit(xs, ts, mode, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad)
    model.reset_state()
    ppl = eval_perplexity(model, corpus)
    print('valid perplexity: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()

    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)
    if(flag == 1):
        break
t2 = time.time()

'''
eval_interval : そこで設定したイテレーションおきにパープレキシティを評価
'''
trainer.plot(ylim=(0, 500)) #グラフの描画

#テストデータで評価
'''
学習終了後にテストデータを使用してパープレキシティを評価
'''
'''
model.reset_state() #LSTMの隠れ状態と記憶セルをリセット
ppl_test = eval_perplexity(model, corpus)
print('test perplexity: ', ppl_test)
'''
#パラメータの保存
model.save_params() #学習後のパラメータを外部ファイルに保存
print("測定時間：%.2f" % (t2-t1))


