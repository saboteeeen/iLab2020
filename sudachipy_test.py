from sudachipy import tokenizer

tokenizer_obj = dictionary.Dictionary(settings).create()
print(type(tokenizer_obj))
# <class 'sudachipy.tokenizer.Tokenizer'>

text = '友人・我孫子とスカイツリーでスパゲティを食った。'

tokens = tokenizer_obj.tokenize(tokenizer.Tokenizer.SplitMode.C, text)
print(type(tokens))
# <class 'sudachipy.morphemelist.MorphemeList'>

for t in tokens:
    print(t.surface(), t.part_of_speech(), t.reading_form(), t.normalized_form())