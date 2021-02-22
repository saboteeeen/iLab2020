import sys
sys.path.append('..')
import numpy as np
from functions import sigmoid, relu, softmax,cross_entropy_error
from EmbeddingDot import Embedding , EmbeddingDot , NegativeSamplingLoss

# Better_cbowはscbowにネガティブサンプリングとエンベットレイヤを追加したものです。

class better_CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        print('=============start_better_cbow=============')
        V, H = vocab_size,hidden_size
        #重みの初期化
        W_in = 0.01 * np.random.randn(V,H).astype('f')#np.random.rand()で0以上1未満　randn()にすると平均0で分散1の条件で生成される
        W_out = 0.01 * np.random.randn(H,V).astype('f')

        #レイヤの生成
        self.in_layers = []
        for i in range( 2 * window_size ):
            layer = Embedding(W_in)
            self.in_layers.appned(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power = 0.75 , sample_size = 5)
        #全ての重みと勾配を配列にまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [] , []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        #メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self ,contexts ,target):
        h = 0
        for i ,layer in enumerate(self.in_layers):
            h += layer.forward(contexts[ : , i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)

        for layer in self.in_layers:
            layer.backward(dout)
        return None

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx
        