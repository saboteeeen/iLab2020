# coding: utf-8
import sys
sys.path.append('..')
from common import config
config.GPU = True
from common.np import *
from rnnlm_gen import BetterRnnlmGen
#import txt
import train_better_lstm
from tangoatume import preprocess


corpus, word_to_id, id_to_word = train_better_lstm.corpus, train_better_lstm.word_to_id, train_better_lstm.id_to_word
print(word_to_id) 
vocab_size = len(word_to_id)
corpus_size = len(corpus)


model = BetterRnnlmGen()
model.load_params('BetterLstmlm.pkl')

# start文字とskip文字の設定
start_word = 'コロナウイルス'
start_id = word_to_id[start_word]
skip_words = ['']
skip_ids = [word_to_id[w] for w in skip_words]
# 文章生成
word_ids = model.generate(start_id, skip_ids)
text = ''.join([id_to_word[i] for i in word_ids])
text = text.replace(' <eos>', '.\n')

print(text)


model.reset_state()

start_words = 'コロナウイルス'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
text = ''.join([id_to_word[i] for i in word_ids])
text = text.replace(' <eos>', '.\n')
print('-' * 50)
print(text)