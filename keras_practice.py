import sys
sys.path.append('..')
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import re
from sudachipy import tokenizer
from sudachipy import dictionary
def preprocess(file):	
	text = sudachi(file) #分かち書きしていないテキストを使う場合はこの行を実行する
	#text = file.read() #分かち書き済みのテキストを使う場合はこの行を実行する
	#print(text)
	text = text.replace('。','　。')
	text = text.replace('\u3000','')
	text = text.replace('\n','')
	words = text.split(' ')
	print(words)

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

	return corpus , word_to_id , id_to_word ,words 

def sudachi(file):
	text = file.read()
	#file = text
	tokenizer_obj = dictionary.Dictionary().create()
	mode = tokenizer.Tokenizer.SplitMode.C
	return " ".join( [m.surface() for m in tokenizer_obj.tokenize(text, mode)] )

def one_hot_v(corpus,vocab_size):
	#one-hot表現への変換
	#:param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
	#:param vocab_size: 語彙数
	#:return: one-hot表現（2次元もしくは3次元のNumPy配列）
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

#学習用テキストデータの読み込み
file_name = "./txt/koi_z.txt"
file = open(file_name,encoding = 'utf-8')

#分かち書きとかリストアップとか
corpus, word_to_id , id_to_word ,words = preprocess(file)
vocab_size = len(word_to_id)
one_hot = one_hot_v(corpus, vocab_size)
size = np.shape(one_hot)
print(vocab_size)
print(np.shape(corpus))

one_hot_reshape = one_hot.reshape(2573,589,1)#高さ,列,行
#重みは勝手に決まる。
model = Sequential()
model.add(Embedding(vocab_size,10))
model.add(LSTM(100, activation = 'tanh' ,dropout = 0.2,))#(hidden_size , )
model.compile( loss = "sparse_categorical_crossentropy" , optimizer = "sgd" )#categorical_crossentropyと同じですが，スパースラベルを取る点で違います
model.summary()#モデルの保存？？

#学習
model.fit(  one_hot_reshape[0, :, :], one_hot_reshape[1, :, :],
			epochs = 100,
			verbose = 2,
			)

#データの保存と読み込み
model.save("model.h5")
load_model = load_model("model.h5")






