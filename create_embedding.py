import os
import time
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re
import string

df = pd.read_excel('icmyc_complaints_IIT_mumbai.xlsx',index=False)
df = df[['description','category_title']]
df['description'] = df['description'].astype(str)
df['category_title'] = df['category_title'].astype(str)
print(df.columns)
print(df.shape)

'''
replace punctuations with white space, re.sub is used to replace more than 1 white space with a single white space only.
'''
df['description']=df['description'].apply(lambda x: x.lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))) 
df['description']=df['description'].apply(lambda x:re.sub(r"\s+", ' ', x))
df['category_title']=df['category_title'].apply(lambda x: x.lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))) 
df['category_title']=df['category_title'].apply(lambda x:re.sub(r"\s+", ' ', x))

df.to_csv('check.csv')
print(df.head(3))

# split sentences into a list of words to create embeddings
sent_list=[]

sent_list = list(df['description'].values)
for i in range(len(sent_list)):
    temp =[]
    temp= sent_list[i].split()
    # temp = set(temp)
    # for word in temp.copy():
    #     if word in stopwords.words('english'):
    #         temp.remove(word)
    # temp=list(temp)
    sent_list[i]=temp
    del temp

cat_list=[]
cat_list = list(df['category_title'].unique())
for i in range(len(cat_list)):
    temp =[]
    temp= cat_list[i].split()
    # temp = set(temp)
    # for word in temp.copy():
    #     if word in stopwords.words('english'):
    #         temp.remove(word)
    # temp=list(temp)
    sent_list.append(temp)
    del temp

'''
using gensim Word2Vec model to create custom embeddings of dim=300
'''
st = time.time()
model = Word2Vec(sent_list,min_count=1,size=300,workers=2, iter=30)
et = time.time()
s = 'Word embedding created in %f secs.' % (et-st)
print(s)

# save to pickle file for later use
# model.wv.save_word2vec_format('icmc_word_vectors.bin')
model.save('icmc_word_vectors.bin')
model.save('icmc_check.pickle')

'''
save learnt embeddings as np.array
'''

wordmodel = Word2Vec.load('icmc_word_vectors.bin')
check=np.zeros(shape=(len(wordmodel.wv.vocab),300))
print(type(check))
count = 0
for word in wordmodel.wv.vocab:
    check[count] = wordmodel[word]
    count+=1
word_vectors = np.asarray(check).astype(np.float32)
print(word_vectors.shape)

with open('icmc_vectors_array.pickle', 'wb') as f:
    pickle.dump(word_vectors, f)


'''
visualize embeddings created
'''
# fit a 2d PCA model to the vectors
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)

# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show() 


    
