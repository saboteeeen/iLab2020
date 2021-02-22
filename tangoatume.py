import sys
sys.path.append('..')
#import MeCab as M
import re
import numpy as np
from common.util import eval_perplexity, to_gpu , to_cpu
def preprocess(file):	
	#text = sudachi(file) #分かち書きしていないテキストを使う場合はこの行を実行する
	text = file.read() #分かち書き済みのテキストを使う場合はこの行を実行する
	#print(text)
	text = text.replace('。','　。')
	text = text.replace('\u3000','')
	text = text.replace('\n','')
	text = text.replace("'", '')
	text = text.replace(',', '')
	words = text.split(' ')
	#print(words)

	word_to_id = {}
	id_to_word = {}
	
	for word in words:
		if word not in word_to_id:
			new_id = len(word_to_id)
			word_to_id[word] = new_id
			id_to_word[new_id] = word
	#print(word_to_id)
	#print(id_to_word)
	corpus = np.array([word_to_id[w] for w in words])
	#print(corpus.shape)

	return corpus , word_to_id , id_to_word  

'''def sudachi(file):
	text = file.read()
	#file = text
	tokenizer_obj = dictionary.Dictionary().create()
	mode = tokenizer.Tokenizer.SplitMode.C
	return " ".join( [m.surface() for m in tokenizer_obj.tokenize(text, mode)] )'''

'''def owakati(text):
	file = text
	data = file.read()
	wakatigaki = M.Tagger("-Owakati").parse(data)
	#print(wakatigaki)
	return wakatigaki'''

def one_hot_v(corpus,vocab_size):
	'''one-hot表現への変換

	:param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
	:param vocab_size: 語彙数
	:return: one-hot表現（2次元もしくは3次元のNumPy配列）
	'''
	N = corpus.shape[0]

	if corpus.ndim == 1:
		one_hot = np.zeros((N, vocab_size), dtype=np.int32)
		for idx, word_id in enumerate(corpus):
			one_hot[idx, word_id] = 1

	elif corpus.ndim == 2:
		C = corpus.shape[1]
		one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
		for idx_0, word_ids in enumerate(corpus):
			for idx_1, word_id in enumerate(word_ids):
				one_hot[idx_0, idx_1, word_id] = 1

	return one_hot

def create_context_target(corpus,window_size=1):
	'''コンテキストとターゲットの作成

	:param corpus: コーパス（単語IDのリスト）
	:param window_size: ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
	:return:
	'''
	target = corpus[window_size:-window_size]
	contexts = []

	for idx in range(window_size, len(corpus) -window_size):
		cs = []
		for t in range(-window_size, window_size + 1):
			if t == 0:
				continue
			cs.append(corpus[idx + t])
		contexts.append(cs)

	return np.array(contexts), np.array(target)

