# coding: utf-8
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from optimizer import SGD
from simple_rnnlm import SimpleRnnlm
from tangoatume import preprocess, create_context_target, one_hot_v
#今回は日本語のデータを使うのでimport dataset は不要

#ハイパーパラメータの設定
batch_size = 10
wordvec_size = 100
hidden_size = 100 #RNNの隠れ状態ベクトルの要素数
time_size = 5 #Truncated BPTTの展開する時間サイズ(どこまで逆伝播で遡るかを決定する)
lr = 0.1
max_epoch = 100

#学習データ
file_name = "./txt/kokoro.txt"
file = open(file_name,encoding='utf-8')
corpus , word_to_id , id_to_word = preprocess(file)

corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # 入力
ts = corpus[1:]  # 出力（教師ラベル）
data_size = len(xs)
print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

#学習時に使用する変数
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

#モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

#ミニバッチの各サンプルの読み込み開始位置を計算
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

#file_name = "RNN_data.txt"
#file = open(file_name, 'w')

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # ミニバッチの取得
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
	#エポックごとにパープレキシティの評価
    ppl = np.exp(total_loss / loss_count)#エポックごとなのでインデントのタブをepochのfor文に合わせる
    print('|epoch %d| perplexity %.2f' % (epoch+1, ppl))
    #file.write(str(ppl) + '\n')
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

model.save_params()
