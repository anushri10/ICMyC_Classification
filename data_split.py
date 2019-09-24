from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import pickle
import string
import re
from gensim.models import Word2Vec
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# import pickle

# with open("./icmc_word_vectors.pickle", "rb") as myFile:
#     model = pickle.load(myFile)

wordmodel = Word2Vec.load('icmc_word_vectors.bin')
print(len(wordmodel.wv.vocab))

# obtain data and labels
df = pd.read_excel('icmyc_complaints_IIT_mumbai.xlsx',index=False)
df = df[['description','category_title']]
df['description'] = df['description'].astype(str)
df['category_title'] = df['category_title'].astype(str)

# replace punctuations with white space, re.sub is used to replace more than 1 white space with a single white space only.
df['description']=df['description'].apply(lambda x: x.lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))) 
df['description']=df['description'].apply(lambda x:re.sub(r"\s+", ' ', x))
df['category_title']=df['category_title'].apply(lambda x: x.lower().translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))) 
df['category_title']=df['category_title'].apply(lambda x:re.sub(r"\s+", ' ', x))

'''
# save category list
cat_df = df[['category_title']]
print(len(cat_df['category_title'].unique()))
pd.DataFrame(cat_df['category_title'].unique()).to_csv('category_list.csv',encoding='utf-8')

# Check if categories lie in custom embdedding space
df_temp = pd.read_csv('category_list.csv',encoding='utf-8')
cat_list = df_temp['0'].tolist()
embed=[]
for single_cat in cat_list:
    embed.append(model[single_cat.split()])
'''

# train-test split 
dataX = df['description'].tolist()
dataY = df['category_title'].tolist()
print("Length of X and Y: ",len(dataX),len(dataY))
X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.20, random_state=42)
print("Train x, Train y: ",len(X_train),type(X_train),len(Y_train),type(Y_train))
print("Test x, Test y: ",len(X_test),len(Y_test))

'''
create index map of words in a sentence (x_train)
'''
train_sentence_word_map=np.full((len(X_train),1000),37702,dtype=int)
for i in range(0,len(X_train)):
    # temp=np.zeros(shape=(1,1000))
    words=X_train[i].split()
    # if len(temp) >907:
    #     print(len(temp))
    # else:
    for j in range(0,len(words)):
        train_sentence_word_map[i][j]=int(wordmodel.wv.vocab.get(words[j]).index)
    
print(type(train_sentence_word_map))
print(len(train_sentence_word_map))
print(type(train_sentence_word_map[0]))
print(len(train_sentence_word_map[0]))
print(len(train_sentence_word_map[1]))  

with open('x_train_sentences.pkl', 'wb') as f:
    pickle.dump(train_sentence_word_map, f)

'''
create index map of words in a sentence (x_test)
'''

test_sentence_word_map=np.full((len(X_test),1000),37702,dtype=int)
for i in range(0,len(X_test)):
    # temp=np.zeros(shape=(1,1000))
    words=X_test[i].split()
    # if len(words) >1000:
    #     print(len(words))
    # else:
    for j in range(0,len(words)):
        test_sentence_word_map[i][j]=int(wordmodel.wv.vocab.get(words[j]).index)
    
print(type(test_sentence_word_map))
print(len(test_sentence_word_map))
print(type(test_sentence_word_map[0]))
print(len(test_sentence_word_map[0]))
print(len(test_sentence_word_map[1]))  

with open('x_test_sentences.pkl', 'wb') as f:
    pickle.dump(test_sentence_word_map, f)

'''
save train and test sentences as embeddings in pickle files
'''
print("Creating embeddings for training and testing data..")
y_train_embeddings=[]

count=0
temp=[]
# x_test_embeddings=np.zeros(shape=(len(X_test),300))
# count=0
x_test_embeddings=[]
for single_comment in X_test:
    # x_test_embeddings[count]=wordmodel[single_comment.split()]
    # count+=1
    x_test_embeddings.append(wordmodel[single_comment.split()])
# X_test_embeddings = np.asarray(x_test_embeddings).astype(np.float32)

print(len(y_train_embeddings))     
print(len(x_test_embeddings))  

print("Starting to create necessary Pickle files..")
st=time.time()
with open('y_train_index.pkl', 'wb') as f:
    pickle.dump(y_train_embeddings, f)
del f
with open('x_test.pkl', 'wb') as f:
    pickle.dump(x_test_embeddings, f)
et = time.time()
s = 'Pickle files created in %f secs.' % (et-st)
print(s)

'''
testing if pickled files can be loaded
'''
# with open('x_train.pkl', 'rb') as f:
#     mynewlist = pickle.load(f)

'''
save pickle files of train_labels and test_labels
'''
print("Starting to create Label Pickle files..")
st=time.time()
with open('y_train.pkl', 'wb') as f:
    pickle.dump(Y_train, f)
del f
with open('y_test.pkl', 'wb') as f:
    pickle.dump(Y_test, f)
et = time.time()
s = 'Label Pickle files created in %f secs.' % (et-st)
print(s)


'''
one hot encode y_train
'''
print(Y_train[4])
print(len(Y_train))
values = array(Y_train)
# print(len(values))

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# print(onehot_encoded[4])

# perform checks
print(type(onehot_encoded[0]))
print(len(onehot_encoded[0]))
print(type(onehot_encoded))
print(len(onehot_encoded))
onehot_encoded = np.asarray(onehot_encoded).astype(np.int32)
# save pickle file of one_h_y_train
with open('y_train_one_h_labels.pickle', 'wb') as f:
    pickle.dump(onehot_encoded, f)

'''
one hot encode y_test
'''
print(Y_test[4])
print(len(Y_test))
values = array(Y_test)
print(len(values))

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(len(onehot_encoded[0]))
# print(onehot_encoded[4])

# perform checks
print(type(onehot_encoded[0]))
print(len(onehot_encoded[0]))
print(type(onehot_encoded))
print(len(onehot_encoded))
onehot_encoded = np.asarray(onehot_encoded).astype(np.int32)

# save pickle file of one_h_y_test
with open('y_test_one_h_labels.pickle', 'wb') as f:
    pickle.dump(onehot_encoded, f)