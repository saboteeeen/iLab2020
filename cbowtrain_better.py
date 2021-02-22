# coding: utf-8
import sys 
sys.path.append('..')
from trainer import Trainer 
from optimizer import Adam as adam
from SimpleCBOW import SimpleCBOW as scbow
from tangoatume import preprocess, create_context_target, one_hot_v
from util import eval_perplexity, to_gpu ,create_contexts_target, to_cpu
from most_sim import most_similar as ms
from Comatrix import create_co_matrix as co_mat
import config
# GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
# ===============================================
config.GPU = True
# ===============================================
import numpy as np
import pickle
from cbow_better import better_CBOW 

#ハイパーパラメータの設定
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 100

def BcbowTrain(file):
	#データ処理
	corpus , word_to_id , id_to_word = preprocess(file)
	vocab_size = len(word_to_id)

	contexts, target = create_context_target(corpus, window_size)
	
	if config.GPU:
		contexts, target = to_gpu(contexts), to_gpu(target)

	target = one_hot_v(target,vocab_size)
	contexts = one_hot_v(contexts,vocab_size)

	#モデル生成
	model = better_CBOW(vocab_size , hidden_size)
	optimizer = adam()
	train = Trainer(model, optimizer)

	
	train.fit(contexts, target, max_epoch, batch_size)
	train.plot()

	word_vecs = model.word_vecs
	if config.GPU:
		word_vecs = to_cpu(word_vecs)
	params = {}
	params['word_vecs'] = word_vecs.astype(np.float16)
	params['word_to_id'] = word_to_id
	params['id_to_word'] = id_to_word
	pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
	with open(pkl_file, 'wb') as f:
		pickle.dump(params, f, -1)

	'''for word_id, word in id_to_word.items():
		print(word,word_vecs[word_id])

	C = co_mat(corpus,vocab_size,window_size = 1)
	ms('彼女',word_to_id,id_to_word,C,top = 10)'''




