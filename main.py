import tangoatume as tango
import numpy as np
from SimpleCBOW import SimpleCBOW as scbow
from ScbowTrain import ScbowTrain 


if __name__ == '__main__':


	file_name = "./txt/koi_z.txt"
	file = open(file_name,encoding='utf-8')
	ScbowTrain(file)







	
	#corpus , word_to_id , id_to_word = tango.preprocess(file)
	#contexts, target = tango.create_context_target(corpus, window_size = 1)
	#vocab_size = len(word_to_id)
	#seq = tango.one_hot_v(corpus,vocab_size)
	#scbow(vocab_size,hidden_size)
	
