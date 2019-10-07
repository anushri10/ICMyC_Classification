import tensorflow as tf
import pickle
import numpy as np
import time
import os
from numpy import linalg as la
from sklearn.model_selection import train_test_split
from tensorflow.contrib.tensorboard.plugins import projector


result_dir = "./"
model_name = "model.ckpt"
path_to_model = result_dir + model_name

train_attention = True
initialize_random = False
train_we = True


#line_no_image_map = np.loadtxt("../dataset/line_no_image_list.csv",delimiter=",",dtype=int)

# load complaint lables
with open("../dataset/dataset_icmc_only_bang_text/desc+title/one_h_lables.pickle", "rb") as myFile:
    lables = pickle.load(myFile,encoding='latin1')

# load sentence index - word index list map.
# It is of format (1,[5,7,9]) where 1 is the sentece index and [5,7,9] is list of word in the sentence.
with open("../dataset/dataset_icmc_only_bang_text/desc+title/sentence_list_of_word_index_map_icmc_padded.pickle", "rb") as myFile:
    sentence_words_index_list_map = pickle.load(myFile,encoding='latin1')

if not initialize_random:

    # load pre-trained word embedding.
    with open("../dataset/dataset_icmc_only_bang_text/desc+title/word_vectors_icmc.pickle", "rb") as myFile:
        word_vectors = pickle.load(myFile,encoding='latin1')

    word_vectors = np.asarray(word_vectors).astype(np.float32)

    for i in range(len(word_vectors) - 1):
        word_vectors[i] /= (la.norm((word_vectors[i])))

#
# for i in range(len(word_vectors) - 1):
#     print np.max(np.abs(word_vectors[i]))

print("Done loading vectors")

# Sentence vectors
dataX = []

# Image vectors
dataY = []



for i in range(len(sentence_words_index_list_map)):

        dataX.append(sentence_words_index_list_map[i])


'''
for i in range(len(lables)):

    dataX.append(sentence_words_index_list_map[i])
    dataY.append(lables[i])
'''
dataX = np.asarray(dataX)
dataY = lables


X_train, X_validation, Y_train, Y_validation = train_test_split(dataX, dataY, test_size=0.15, random_state=42)

#X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.20, random_state=39)


print(("done loading dataX: ", dataX.shape," dataY: ",dataY.shape," X_train: ",X_train.shape," Y_train : ",Y_train.shape," X_validation  : ",X_validation.shape ," Y_validation",Y_validation.shape))

vocab_size = len(word_vectors)
embedding_dim = 300
learning_rate = 1e-3
# decay_factor = 0.99
max_padded_sentence_length = 60
batch_size = 100
iterations = 200
highest_val_acc = 0


def init_weight(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=name,dtype = tf.float32)
    return tf.Variable(initial)


def init_bias(shape,name):
    initial = tf.truncated_normal(shape=shape,stddev=0.1, name=name,dtype=tf.float32)
    return tf.Variable(initial)


if initialize_random:

    # Initial embedding initialized randomly
    embedding_init = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_dim], stddev=0.1, dtype=tf.float32), trainable=train_we, name="word_embedding")

else:

    # Initial embedding initialized by word2vec vectors
    embedding_init = tf.Variable(tf.constant(word_vectors, shape=[vocab_size, embedding_dim]), trainable=train_we, name="word_embedding")

config = projector.ProjectorConfig()


# It will hold tensor of size [batch_size, max_padded_sentence_length]
X = tf.placeholder(tf.int32, [None, max_padded_sentence_length])

# Word embedding lookup
word_embeddings = tf.nn.embedding_lookup(embedding_init, X)

if train_attention:

    in_size = tf.shape(word_embeddings)[0]

    reshaped_w_e = tf.reshape(word_embeddings,[in_size * max_padded_sentence_length, embedding_dim])

    print(reshaped_w_e)

    no_of_nurons_h1 = 512
    Wa = init_weight([embedding_dim,no_of_nurons_h1],'Wa')
    ba = init_bias([no_of_nurons_h1],'ba')
    ya = tf.nn.relu(tf.nn.dropout((tf.matmul(reshaped_w_e,Wa) + ba),0.5))

    # Hidden layer of size 512
    no_of_nurons_h2 = 512
    Wa1 = init_weight([no_of_nurons_h1,no_of_nurons_h2],'Wa1')
    ba1 = init_bias([no_of_nurons_h2],'ba1')
    ya1 = tf.nn.relu(tf.nn.dropout((tf.matmul(ya,Wa1) + ba1),0.5))

    Wa2 = init_weight([no_of_nurons_h2,1],'Wa2')
    ba2 = init_bias([1],'ba2')

    # Output layer of the neural network.
    ya2 = tf.matmul(ya1,Wa2) + ba2

    attention_reshaped = tf.reshape(ya2, [in_size, max_padded_sentence_length])

    attention_softmaxed = tf.nn.softmax(attention_reshaped)

    attention_expanded = tf.expand_dims(attention_softmaxed, axis=2)


    # Attention based weighted averaging of word vectors.
    sentence_embedding = tf.reduce_sum(tf.multiply(word_embeddings, attention_expanded),axis=1)

else:

    # Simply Average out word embedding to create sentence embedding
    sentence_embedding = tf.reduce_mean(word_embeddings, axis=1)

embedding_tb = config.embeddings.add()
embedding_tb.tensor_name = embedding_init.name

# Linking word vectors with metadata
embedding_tb.metadata_path = os.path.join(result_dir, 'metadata.tsv')


def get_batches(X, Y, bsize):
    for i in range(0, len(X) - bsize + 1, bsize):
        indices = slice(i, i + bsize)
        yield X[indices], Y[indices]


input_layer_size = embedding_dim
output_layer_size = len(lables[0])


# Hidden layer of size 1024
no_of_nurons_h1 = 1024
W = init_weight([input_layer_size,no_of_nurons_h1],'W')
b = init_bias([no_of_nurons_h1],'b')
y = tf.nn.relu(tf.nn.dropout((tf.matmul(sentence_embedding,W) + b),0.5))

# Hidden layer of size 1024
no_of_nurons_h2 = 1024
W1 = init_weight([no_of_nurons_h1,no_of_nurons_h2],'W1')
b1 = init_bias([no_of_nurons_h2],'b1')
y1 = tf.nn.relu(tf.nn.dropout((tf.matmul(y,W1) + b1),0.5))

W2 = init_weight([no_of_nurons_h2,output_layer_size],'W2')
b2 = init_bias([output_layer_size],'b2')

# Output layer of the neural network.
y2 = tf.matmul(y1,W2) + b2

# It will hold the true label for current batch
y_ = tf.placeholder(tf.int32, shape=[None,output_layer_size])

check_op = tf.add_check_numerics_ops()


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y2,labels=y_))

# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


variables_names = [v.name for v in tf.trainable_variables()]


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter(result_dir)
projector.visualize_embeddings(summary_writer, config)


saver = tf.train.Saver()


print("Training...")

#values = sess.run(variables_names)

# for k, v in zip(variables_names, values):
#     print "Variable: ", k
#     print "Shape: ", v.shape
#     # print v



log = open(result_dir + 'log', 'w')
log.write('#Index, Training Time, Training Loss, Validation Loss, Train Acc., Validation Acc.\n')
log.close()


for i in range(iterations):
    print(("Iteration: " + str(i + 1)))
    start = time.time()
    training_error = 0
    validation_error = 0
    t_count = 0
    v_count = 0
    t_acc = 0
    v_acc = 0;

    for batch in get_batches(X_train, Y_train, batch_size):
        batch_xs, batch_ys = batch
        feed_dict = {X: batch_xs, y_: batch_ys }
        t_e, _, _= sess.run([loss,train_step,check_op], feed_dict=feed_dict)
        t_acc += sess.run(accuracy, feed_dict={X: batch_xs, y_: batch_ys})
        training_error += t_e
        t_count += 1

    for batch in get_batches(X_validation, Y_validation, batch_size):
        batch_xs, batch_ys = batch
        feed_dict = {X: batch_xs, y_: batch_ys}
        v_acc += sess.run(accuracy, feed_dict={X: batch_xs, y_: batch_ys})
        v_count += 1
        validation_error += loss.eval({X: batch_xs, y_: batch_ys}, session=sess)

    training_time = str( time.time() - start)
    training_error /= t_count
    validation_error /= v_count

    t_acc /= t_count
    v_acc /= v_count

    # Apply Learning rate decay for SGD.
    # learning_rate = learning_rate * decay_factor

    print(("Time : "+str(training_time) + " Seconds" + " training_error : " + str(training_error) + " validation_error : "+ str(validation_error)))
    print("Train acc : ", t_acc*100," Validation acc : ", v_acc*100)
    log = open(result_dir + 'log', 'a')
    log.write(str(i + 1) + ' ' + training_time + ' ' + str(training_error)+' ' + str(validation_error)+' ' + str(t_acc * 100)+' ' + str(v_acc * 100)+'\n')
    log.close()

    if v_acc > highest_val_acc:
        highest_val_acc = v_acc
        save_path = saver.save(sess, path_to_model)
        print(("Model saved in file : %s" % save_path))
        fail_count = 0


print("Done Training :)")
'''
#Restore the best model to calculate the test accuracy.
saver.restore(sess,path_to_model)

print "Test Acc : ", sess.run(accuracy, feed_dict={X:X_test,y_:Y_test}) * 100

print("Done Training :)")
print("Model saved in file : %s" % save_path)

'''

