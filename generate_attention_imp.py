# coding: utf-8
import sys
sys.path.append('..')
from common import config
config.GPU = False
from common.np import *
import matplotlib.pyplot as plt
from tangoatume import preprocess 
from common.optimizer import Adam
from common.trainer1 import RnnlmTrainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
from attention_seq2seq import AttentionSeq2seq
import time
from common.util import eval_perplexity, to_gpu
# データセットの読み込み
file_name = "./text/owakachi.txt"
file = open(file_name,encoding='utf-8')
corpus, word_to_id, id_to_word = preprocess(file)

if config.GPU:
    corpus = to_gpu(corpus)

xs = corpus[:-1]
ts = corpus[1:]


#print(xs.shape)
# ハイパーパラメータの設定
vocab_size = len(word_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 1
max_epoch = 50
max_grad = 5.0
sample_size = 100
lr = 0.001
time_size = 35

#モデルの生成
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = RnnlmTrainer(model, optimizer)

#学習
best_ppl = float('inf')
t1 = time.time()
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                max_grad=max_grad)

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
t2 = time.time()
print("測定時間：%.2f" % (t2-t1))
#学習終了
# start文字とskip文字の設定
start_word = 'コロナウイルス'
start_id = word_to_id[start_word]
skip_words = ['']
skip_ids = [word_to_id[w] for w in skip_words]

batch_x, batch_t = trainer.get_batch(xs, ts, batch_size, time_size)
# 文章生成
'''
x = np.array([start_id]).reshape((1, 1))
hs = model.encoder.forward(x)
print(hs.shape)
'''

word_ids = model.generate(start_id, batch_x,sample_size)
print(type(word_ids))
print(type(id_to_word))
new_list = []
for i in word_ids:
    new_list.append(int(i))
'''
while len(word_to_id) < sample_size:
    x = sampled
    word_ids.append(int(x))
'''
#word_ids = start_ids[:-1] + word_ids

text = ''.join([id_to_word[i] for i in new_list])
text = text.replace(' <eos>', '.\n')
print('-' * 50)
print(text)




