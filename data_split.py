from sklearn.model_selection import train_test_split
import time
import pandas as pd
import pickle
import string
import re
from gensim.models import Word2Vec
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
print("Train x, Train y: ",len(X_train),len(Y_train))
print("Test x, Test y: ",len(X_test),len(Y_test))

print("Creating embeddings for training and testing data..")
# save train and test sentences as embeddings in pickle files
x_train_embeddings=[]
for single_comment in X_train:
    x_train_embeddings.append(wordmodel[single_comment.split()])
x_test_embeddings=[]
for single_comment in X_test:
    x_test_embeddings.append(wordmodel[single_comment.split()])

print(len(x_train_embeddings))     
print(len(x_test_embeddings))  

print("Starting to create necessary Pickle files..")
st=time.time()
with open('x_train.pkl', 'wb') as f:
    pickle.dump(x_train_embeddings, f)
del f
with open('x_test.pkl', 'wb') as f:
    pickle.dump(x_test_embeddings, f)
et = time.time()
s = 'Pickle files created in %f secs.' % (et-st)
print(s)

'''
# testing if pickled files can be loaded
with open('x_train.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
'''
