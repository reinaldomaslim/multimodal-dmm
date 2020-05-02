import spacy
import numpy as np 

nlp = spacy.load("en_core_web_md")
tokens = nlp("dog cat banana afskfsd, banana, Banana, banana")


for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))