# coding: utf-8
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
import train_custom_loop

corpus, word_to_id, id_to_word = train_custom_loop.corpus , train_custom_loop.word_to_id , train_custom_loop.id_to_word
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen(vocab_size , train_custom_loop.wordvec_size, train_custom_loop.hidden_size )
model.load_params('SimpleRnnlm.pkl')

# start文字とskip文字の設定
start_word = '私'
start_id = word_to_id[start_word]
skip_words = ['（','）']


skip_ids = [word_to_id[w] for w in skip_words]
# 文章生成
word_ids = model.generate(start_id, skip_ids)
print(word_ids)
txt = ''.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '。\n')
print(txt)