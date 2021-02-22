import sys 
sys.path.append('..')
from trainer import Trainer 
from optimizer import Adam as adam
from SimpleCBOW import SimpleCBOW as scbow
from tangoatume import preprocess, create_context_target, one_hot_v
from util import eval_perplexity, to_gpu
from most_sim import most_similar as ms
from Comatrix import create_co_matrix as co_mat
#ハイパーパラメータの設定
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 100

def ScbowTrain(file):
	corpus , word_to_id , id_to_word = preprocess(file)

	contexts, target = create_context_target(corpus, window_size = 1)
	vocab_size = len(word_to_id)


	target = one_hot_v(target,vocab_size)
	contexts = one_hot_v(contexts,vocab_size)

	model = scbow(vocab_size , hidden_size)
	optimizer = adam()
	train = Trainer(model, optimizer)



	
	train.fit(contexts, target, max_epoch, batch_size)
	train.plot()

	word_vecs = model.word_vecs

	for word_id, word in id_to_word.items():
		print(word,word_vecs[word_id])

	C = co_mat(corpus,vocab_size,window_size = 1)
	ms('彼女',word_to_id,id_to_word,C,top = 10)




