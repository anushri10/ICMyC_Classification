import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file_path = r"D:\ICMC_new\updated_new_work"

def label_distribution(train_y):
    class_count=[None]*42
    for single in train_y:
        a = np.argmax(single)
        if class_count[a]==None:
            class_count[a]=1
        else:
            class_count[a]+=1

    x = [i for i in range(0,42)]


    print(len(class_count))
    print(len(x))
    plt.bar(x,class_count,align='center') # A bar chart
    plt.xlabel('labels')
    plt.ylabel('Frequency')
    plt.show()
    return class_count


with open(file_path + "/sentence_word_index_train.pickle", "rb") as myFile:
    train_x = pickle.load(myFile, encoding='latin1')

with open(file_path + "/y_train_labels.pickle", "rb") as myFile:
            # OF NO USE.. as this is unsupervised domain adaptation
    train_y = pickle.load(myFile, encoding='latin1')

# print(len(train_y))
# print(train_y[0])

# label_distribution(train_y)


X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.15, stratify=train_y , random_state=22)

print(len(X_train))
print(len(Y_train),len(Y_test))
class_count= label_distribution(Y_train)
print(class_count)

print('##########\n\n')
print("TEST.\n")
class_count_2= label_distribution(Y_test)
print(class_count_2)


with open(file_path + "/y_test_labels.pickle", "rb") as myFile:
    original_test_y= pickle.load(myFile, encoding='latin1')
with open(file_path + "/sentence_word_index_test.pickle", "rb") as myFile:
    original_test_x = pickle.load(myFile, encoding='latin1')

print("&&&&\n\n")
x_test_final = np.append(X_test,original_test_x,axis=0)
y_test_final = np.append(Y_test,original_test_y,axis=0)

save_path = r"D:\ICMC_new\updated_new_work\updated_input"
with open(save_path + "/sentence_word_index_train_update.pickle", "wb") as myFile:
    pickle.dump(X_train,myFile)

with open(save_path + "/y_train_labels_update.pickle", "wb") as myFile:
    pickle.dump(Y_train,myFile)

with open(save_path + "/sentence_word_index_test_update.pickle", "wb") as myFile:
    pickle.dump(x_test_final,myFile)

with open(save_path + "/y_test_labels_update.pickle", "wb") as myFile:
    pickle.dump(y_test_final,myFile)

