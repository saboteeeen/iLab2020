import spacy
 
def main():
  nlp = spacy.load('ja_ginza_nopn')
  doc = nlp('依存構造解析の実験を行っています。')
  for sent in doc.sents:
    for token in sent:
      print(token.i, token.orth_, token.lemma_, token.pos_, token.dep_, token.head.i)
    print('EOS')
 
# 直接実行されたときはmain関数を呼び出す
if __name__ == "__main__":
  main()