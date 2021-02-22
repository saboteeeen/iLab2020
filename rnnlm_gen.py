import sys
sys.path.append('..')
from common.np import *
from functions import softmax
from simple_rnnlm import SimpleRnnlm
from simple_Lstmlm import SimpleLstmlm
from better_lstm import BetterLstmlm

class RnnlmGen(SimpleLstmlm):
	def generate(self, start_id, skip_ids=None, sample_size=100): #start_id:最初に与える単語ID sample_size:サンプリングする単語の数 skip_ids:ここで指定された単語IDはサンプリングされない
		word_ids = [start_id]

		x = start_id
		#print(type(start_id))
		while len(word_ids) < sample_size:
			x = np.array(x).reshape(1, 1) #入力xは二次元配列
			score = self.predict(x) #各単語のスコアを出力
			p = softmax(score.flatten()) #スコアをsoftmaxで正規化、pは確率分布

			sampled = np.random.choice(len(p), size=1, p=p) #確率分布からのサンプリング
			if(skip_ids is None) or (sampled not in skip_ids):
				x = sampled
				word_ids.append(int(x))

		return word_ids

	def get_state(self):
		return self.lstm_layer.h, self.lstm_layer.c

	def set_state(self, state):
		self.lstm_layer.set_state(*state)

class BetterRnnlmGen(BetterLstmlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)