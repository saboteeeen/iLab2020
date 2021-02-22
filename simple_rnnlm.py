import sys
import pickle 
sys.path.append('..')
from common.np import *  # import numpy as np (or import cupy as np)
#import numpy as np
from common.time_layers import *

class SimpleRnnlm:
	def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100): #必要なパラメータを初期化し、レイヤを生成
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn

		#重みの初期化
		embed_W = (rn(V, D) / 100).astype('f')
		rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
		rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
		rnn_b = np.zeros(H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')
		#Xavierの初期値(1/sqrt(n)の標準偏差を持つ分布)

		#レイヤの生成
		self.layers = [
			TimeEmbedding(embed_W),
			TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
			TimeAffine(affine_W, affine_b)
		] #Truncated BPTTで学習することを想定:stateful=True → 前時刻の隠れ状態を引き継げる 
		self.loss_layer = TimeSoftmaxWithLoss()
		self.rnn_layer = self.layers[1]

		#全ての重みと勾配をリストにまとめる
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def predict(self, xs): #softmaxレイヤの直前までを処理する
		for layer in self.layers:
			xs = layer.forward(xs)
		return xs

	def forward(self, xs, ts):
		for layer in self.layers:
			xs = layer.forward(xs)
		loss = self.loss_layer.forward(xs, ts)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def resrt_state(self): #ネットワークの状態をリセット
		self.rnn_layer.resrt_state()

	def save_params(self, file_name='SimpleRnnlm.pkl'):
		with open(file_name, 'wb') as f:
			pickle.dump(self.params, f)

	def load_params(self, file_name='SimpleRnnlm.pkl'):
		with open(file_name, 'rb') as f:
			self.params = pickle.load(f)