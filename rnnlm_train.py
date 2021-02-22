import sys
sys.path.append('..')
from common import config
config.GPU = True
import matplotlib.pyplot as plt
#import numpy as np
from common.np import *  # import numpy as np (or import cupy as np)
from optimizer import SGD
import text
from simple_rnnlm import SimpleRnnlm
from tangoatume import preprocess
from common.util import eval_perplexity, to_gpu
import time
import pandas as pd
import openpyxl

#ハイパーパラメータの設定
batch_size = 20
wordvec_size = 650
hidden_size = 650 #RNNの隠れ状態ベクトルの要素数
time_size = 35 #Truncated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 300

#学習データの読み込み(データセットを小さくする)
file_name = "./text/owakachi.txt"
file = open(file_name,encoding='utf-8')
corpus, word_to_id, id_to_word = preprocess(file)
if config.GPU:
    corpus = to_gpu(corpus)
#corpus_size = 1000
#corpus = corpus[:corpus_size]
#vocab_size = int(max(corpus) + 1
corpus_size = len(corpus)
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print('corpus_size: %d, vocabulary size: %d' % (corpus_size, vocab_size))
#print('vocabulary size: %d' % (vocab_size))

#学習時に使用する変数
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

#モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

#ミニバッチの各サンプル
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)] #offsetsの各要素には、データの読み込む開始位置が格納

#データ収集を行うための準備
file_name = "RNN_data.txt"
file = open(file_name, 'w')

t1 = time.time()
for epoch in range(max_epoch):
	for iter in range(max_iters):
		#ミニバッチの取得
		batch_x = np.empty((batch_size, time_size), dtype='i') #batch_xという容器を用意
		batch_t = np.empty((batch_size, time_size), dtype='i') #batch_tという容器を用意
		for t in range(time_size): #time_idxを順に増やしながらtime_idxの場所のデータをコーパスから取得
			for i, offset in enumerate(offsets):
				#各バッチでオフセットを加える
				batch_x[i, t] = xs[(offset + time_idx) % data_size] #コーパスを読み込む場所がコーパスサイズを超えた場合に、コーパスの先頭に戻す(コーパスのサイズで割った余りをインデックスとして使用)
				batch_t[i, t] = ts[(offset + time_idx) % data_size]
			time_idx += 1

		#勾配を求め、パラメータを更新
		loss = model.forward(batch_x, batch_t)
		model.backward()
		optimizer.update(model.params, model.grads)
		total_loss += loss
		loss_count += 1

	#パープレキシティの計算
	ppl = np.exp(total_loss / loss_count) #エポックごとのパープレキシティを求める
	print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
	file.write(str(ppl) + '\n')
	ppl_list.append(float(ppl))
	total_loss, loss_count = 0, 0
model.save_params()
t2 = time.time()
print("測定時間：%.2f" % (t2-t1))

df1 = pd.DataFrame([ppl_list])
df1.index = ['ppl']
#with pd.ExcelWriter('kekka.xlsx',mode ='a') as writer:
df1.to_excel('./rnnlm300.xlsx')
