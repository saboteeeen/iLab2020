# coding: utf-8
import sys
sys.path.append('..')
from common import config
config.GPU = True
from rnnlm_gen import RnnlmGen
#import rnnlm_train
#import train_better_lstm
import train_simple_lstmlm
#import txt
#from tangoatume import preprocess

#file_name = "./txt/koi_z.txt"
#file = open(file_name,encoding='utf-8')
corpus, word_to_id, id_to_word = train_simple_lstmlm.corpus, train_simple_lstmlm.word_to_id, train_simple_lstmlm.id_to_word
#print(word_to_id) 
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen(vocab_size, train_simple_lstmlm.wordvec_size, train_simple_lstmlm.hidden_size)
model.load_params("./pkl/SimpleLstmlm_epoch50.pkl")

#start文字とskip文字の設定
start_word = 'コロナウイルス'
start_id = word_to_id[start_word]
skip_words = ['']
skip_ids = [word_to_id[w] for w in skip_words]

#文章生成
word_ids = model.generate(start_id, skip_ids)
#print(word_ids)
'''
map_result = map(str, word_ids)
result = ' '.join(map_result)
for i in result
'''
text = ''.join([id_to_word[i] for i in word_ids])
text = text.replace(' <eos>', '。\n')
print(text) 