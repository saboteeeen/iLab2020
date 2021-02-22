# coding: utf-8
import sys
sys.path.append('..')
from common import config
config.GPU = True
from common.optimizer import SGD
from common.trainer2 import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from better_lstm import BetterLstmlm
from tangoatume import preprocess
import time

#ハイパーパラメータの設定

batch_size = 20
wordvec_size = 200
hidden_size = 200 #RNNの隠れ状態ベクトルの要素数
time_size = 35 #RNNを展開するサイズ
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

#学習データの読み込み
order_average = float(input("delta_average?:"))

file_name = "./text/owakachi.txt"
file = open(file_name,encoding='utf-8')
corpus, word_to_id, id_to_word = preprocess(file)

if config.GPU:
    corpus = to_gpu(corpus)

#corpus_test = preprocess(file)
vocab_size = len(word_to_id)
#print(vocab_size)

#print(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

#モデルの生成
model = BetterLstmlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer) #モデルの学習
best_ppl = float('inf')
t1 = time.time()
for epoch in range(max_epoch):
    delta_average = trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad)
    model.reset_state()
    ppl = eval_perplexity(model, corpus)
    print('valid perplexity: ', ppl)
    print("="*50)#
    print('ppl_delta_ave = %f' % (delta_average))#
    print("="*50)#
    print('order_ave = %f' % (order_average))
    
    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr
    if delta_average < order_average:
        break

    """if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()"""
    
    model.reset_state()
    #print('-' * 50)
t2 = time.time()
print("測定時間：%.2f" % (t2-t1))