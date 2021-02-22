# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from tangoatume import preprocess 
from dataset import sequence
from common.optimizer import Adam
from common.trainer import RnnlmTrainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
import time

mode = int(input("収束条件有りなら1、学習回数決め打ちは0:"))
file_name = "./text/owakachi.txt"
file = open(file_name,encoding='utf-8')
corpus, word_to_id, id_to_word = preprocess(file)

xs = corpus[:-1]
ts = corpus[1:]

#print(len(word_to_id)) 10582

'''(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()'''
#print(len(char_to_id)) 13

#print(x_train.shape) 45000,7
#print(t_train.shape) 45000,5

# Reverse input? =================================================
is_reverse = False #True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

# ハイパーパラメータの設定
vocab_size = len(word_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 1
max_epoch = 25
max_grad = 5.0

# Normal or Peeky? ==============================================
#model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
# ================================================================
optimizer = Adam()
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
t1 = time.time()
for epoch in range(max_epoch):
    flag = trainer.fit(xs, ts, mode, max_epoch=1, batch_size=batch_size,
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
    if(flag == 1):
        break
t2 = time.time()
print("測定時間：%.2f" % (t2-t1))





'''acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

# グラフの描画
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()'''

