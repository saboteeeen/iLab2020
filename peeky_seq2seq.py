# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder
from common.np import *  # import numpy as np
import pickle


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            p = softmax(score.flatten())
            sample_id = np.random.choice(len(p), size=1, p=p)
            sampled.append(sample_id)

        return sampled

#=======================================================================
class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        self.lstm_layers = []
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        self.enlayers = [self.encoder.embed,
                         self.encoder.lstm]
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def reset_state(self):
        self.encoder.lstm.reset_state()
        self.decoder.lstm.reset_state()
        
    def save_params(self, file_name='PeekySeq2seq.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='PeekySeq2seq.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
    def generate(self, start_id, xs ,skip_ids=None, sample_size=100):
        count = 1
        word_ids = [start_id]
        x = start_id
        hs = self.encoder.forward(xs)#batch_xを入力
        x = np.array(x).reshape(1, 1)
        #hs = hs.reshape(1,self.hidden_size)
        sampled = self.decoder.generate(hs, x, sample_size)#100words returned
        while count < len(sampled):
            if (skip_ids is None) or (sampled != skip_ids):
                x = sampled
                word_ids.append(int(x[count]))# int function can not use type of list as input. 
            count += 1
        return word_ids
    def enpredict(self, xs): #Softmaxレイヤの直前までを処理する(文章生成で使用)
        for layer in self.enlayers:
            xs = layer.forward(xs)
        return xs

#=======================================================================
#=======================================================================

class Better_Peeky_Seq2seq():
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        #初期化
        self.lstm_layers = []
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.hidden_size = hidden_size

        self.encoder = Better_Encoder(V, D, H)
        self.decoder = Better_Peeky_Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        """self.enlayers = [self.encoder.embed,
                         self.encoder.lstm1,
                         self.encoder.lstm2,]"""

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
    
    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)# 1, 35
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def reset_state(self):
        self.lstm_layers = self.encoder.lstm1.reset_state()
        self.lstm_layers = self.encoder.lstm2.reset_state()
        self.lstm_layers = self.decoder.lstm1.reset_state()
        self.lstm_layers = self.decoder.lstm2.reset_state()

    def save_params(self, file_name='BetterPeekySeq2seq.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='BetterPeekySeq2seq.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
    def generate(self, start_id, xs,skip_ids=None, sample_size=100):   
        count = 1 
        word_ids = [start_id]
        x = start_id
        hs = self.encoder.forward(xs)#batch_xを入力
        x = np.array(x).reshape(1, 1)
        sampled = self.decoder.generate(hs, x, sample_size)
        while count < len(sampled):
            if (skip_ids is None) or (sampled != skip_ids):
                x = sampled
                word_ids.append(int(x[count]))# int function can not use type of list as input. 
            count += 1

        return word_ids
    """def enpredict(self, xs): #Softmaxレイヤの直前までを処理する(文章生成で使用)
        for layer in self.enlayers:
            xs = layer.forward(xs)
        return xs"""


class Better_Encoder():
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        
        #初期化
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')

        #レイヤの生成
        self.embed = TimeEmbedding(embed_W)
        #self.drop1 = TimeDropout(dropout_ratio)
        self.lstm1 = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
        #self.drop2 = TimeDropout(dropout_ratio)
        self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)

        self.layers = [ self.embed , self.lstm1 ,  self.lstm2]
        self.lstm_layers = [ self.lstm1 , self.lstm2]
        #self.drop_layers = [ self.drop1 , self.drop2]

        #パラメータの統合
        self.params = self.embed.params + self.lstm1.params + self.lstm2.params
        self.grads = self.embed.grads + self.lstm1.grads + self.lstm2.grads
        self.hs = None

    def forward(self, xs, train_flg=True):

        out = self.embed.forward(xs)
        out = self.lstm1.forward(out)
        h = self.lstm2.forward(out)
        self.h = h
        
        return h[:, -1, :]

    def backward(self, dh):

        dhs1 = np.zeros_like(self.h)
        dhs1[:, -1, :] = dh
        '''
        dhs2 = np.zeros_like(self.h2)
        dhs2[:, -1, :] = dh2
    '''
        dout = self.lstm2.backward(dhs1)
        #dout = self.drop2.backward(dout)
        #dout = dout + dhs1
        dout = self.lstm1.backward(dout)
        #dout = self.drop1.backward(dout)
        dout = self.embed.backward(dout)
        return dout

class Better_Peeky_Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        #初期化
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        self.D = wordvec_size
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H + H, 4 * H) / np.sqrt(H + H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        #レイヤの生成
        self.embed = TimeEmbedding(embed_W)
        #self.drop1 = TimeDropout(dropout_ratio)
        self.lstm1 = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
        #self.drop2 = TimeDropout(dropout_ratio)
        self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)
        #self.drop3 = TimeDropout(dropout_ratio)
        self.affine = TimeAffine(affine_W, affine_b)

        self.layers = [ self.embed  , self.lstm1 
         , self.lstm2  , self.affine]

        self.params, self.grads = [], []

        self.params = self.embed.params + self.lstm1.params + self.lstm2.params + self.affine.params
        self.grads = self.embed.grads + self.lstm1.grads + self.lstm2.grads + self.affine.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape
        #N, H2 = h2.shape
        #print(type(self.lstm1))
        self.lstm1.set_state(h)
        #self.lstm2.set_state(h2)

        #print(N,T)
        #print(N,H)
        out = self.embed.forward(xs)
        hs1 = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs1, out), axis=2)
        #out = self.drop1.forward(out)
        out = self.lstm1.forward(out)
        out = np.concatenate((hs1, out), axis=2)
        #out = self.drop2.forward(out)
        
        out = self.lstm2.forward(out)
        out = np.concatenate((hs1, out), axis=2)
        #out = self.drop3.forward(out)
        score = self.affine.forward(out)

        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache
        D = self.D
        dout = self.affine.backward(dscore)
        #print(dout.shape)
        dout, dhs0 = dout[:, :, :H], dout[:, :, :H]
        #print(dout.shape,dhs0.shape)
        #dout = self.drop3.backward(dout)
        #print(dout.shape)
        #dout, dhs1 = dout[:, :, :H], dout[:, :, :H]
        #print(dout.shape,dhs0.shape)
        dout = self.lstm2.backward(dout)
        #print(dout.shape)
        dout, dhs2 = dout[:, :, :H], dout[:, :, :H]
        #print(dout.shape,dhs2.shape)
        #dout = self.drop2.backward(dout)
        #print(dout.shape)
        #dout, dhs3 = dout[:, :, :H], dout[:, :, :H]
        #print(dout.shape,dhs3.shape)
        dout = self.lstm1.backward(dout)
        #print(dout.shape)
        dembed, dhs4 = dout[:, :, :D], dout[:, :, :H]
        #print(dout.shape,dhs4.shape)
        #dout = self.drop1.backward(dout)
        #print(dout.shape)
        #dembed, dhs5 = dout[:, :, :D], dout[:, :, :H]
        #print(dembed.shape,dhs5.shape)
        self.embed.backward(dembed)

        
        dh1 = dhs0 + dhs2 + dhs4
        #dh2 = dhs0 + dhs2
        dh1 = self.lstm1.dh + np.sum(dh1, axis=1)
        #dh2 = self.lstm2.dh + np.sum(dh2, axis=1)
        #dh2 = self.lstm2.dh + np.sum(dhs2, axis=1)
        return dh1

    def generate(self, h, start_id, sample_size):
        sampled = [start_id]
        sample_id = start_id
        self.lstm1.set_state(h)
        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm1.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm2.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)
            p = softmax(score.flatten())
            sample_id = np.random.choice(len(p), size=1, p=p)
            sampled.append(sample_id)

        return sampled

        