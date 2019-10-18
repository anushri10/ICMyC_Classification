import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import time
from numpy import linalg as la
import pickle


class CDAN:
    def __init__(self):
        self.result_dir = "./save_test_2_update/"
        self.model_name = "model.ckpt"
        self.path_to_model = self.result_dir + self.model_name
        self.batch_size = 32
        self.iterations = 20

        # self.learning_rate = 1e-4
        self.learning_rate=1e-2
        self.embedding_dim = 300
        # self.no_class_labels = None
        self.trade_off = 0.3
        self.keep_prob=0.6
        self.no_attn_hidden_layer_2 = 32
        self.no_attn_hidden_layer_1 = 64
        self.no_attn_hidden_128 = 128
        self.upper_0 = 448
        self.upper = 384
        # self.upper_2 = 512
        self.no_hidden_layer_1_0 = 256
        self.no_hidden_layer_1_0_0=192
        self.no_hidden_layer_1_1 = 128
        self.no_hidden_layer_1_1_0=96
        self.no_hidden_layer_1 = 64
        self.no_hidden_layer_2 = 32
        self.no_disc_hidden_layer_upper_0 = 448
        self.no_disc_hidden_layer_upper = 384
        self.no_disc_hidden_layer_1_0 = 256
        self.no_disc_hidden_layer_192=192
        self.no_disc_hidden_layer_1_1 = 128
        self.no_disc_hidden_layer_96=96
        self.no_disc_hidden_layer_1 = 64
        self.no_disc_hidden_layer_2 = 32
        self.max_padded_length = 60
        self.no_lstm_units = 256

        self.load_data()
        self.define_model()

    def load_data(self):
        file_path = r"D:\ICMC_new\updated_new_work\updated_input"

        # load pre-trained word embedding.
        with open(file_path + "/word_vectors_icmc.pickle", "rb") as myFile:
            self.word_vectors = pickle.load(myFile, encoding='latin1')

        self.word_vectors = np.asarray(self.word_vectors).astype(np.float32)

        for i in range(len(self.word_vectors) - 1):
            self.word_vectors[i] /= (la.norm((self.word_vectors[i])))

        # load target train sentences and labels
        with open(file_path + "/y_train_labels_update.pickle", "rb") as myFile:
            # OF NO USE.. as this is unsupervised domain adaptation
            self.target_train_label = pickle.load(myFile, encoding='latin1')
        with open(file_path + "/sentence_word_index_train_update.pickle", "rb") as myFile:
            self.target_train = pickle.load(myFile, encoding='latin1')

        # load target test sentences and labels
        with open(file_path + "/y_test_labels_update.pickle", "rb") as myFile:
            self.target_test_label = pickle.load(myFile, encoding='latin1')
        with open(file_path + "/sentence_word_index_test_update.pickle", "rb") as myFile:
            self.target_test = pickle.load(myFile, encoding='latin1')

        # load source sentences and labels
        # with open(file_path + "/one_h_lables.pickle", "rb") as myFile:
        #     self.source_label = pickle.load(myFile, encoding='latin1')
        # with open(file_path + "/source_train_sentence_word_index.pkl", "rb") as myFile:
        #     self.source_train = pickle.load(myFile, encoding='latin1')

        # reduce 16 lakh review text to 1 lakh for fast training
        # index = np.random.choice(self.source_train.shape[0],100000,replace=True)
        # self.source_train = self.source_train[index]
        # self.source_label = self.source_label[index]

        self.vocab_size = len(self.word_vectors)
        self.no_class_labels = 42

        # print("Source Sentence Count: ", len(self.source_train))
        print("Target Sentence Count: ", len(self.target_train))
        print("Target Test Sentence Count: ", len(self.target_test))
        print("Vocab size: ", self.vocab_size)

    def define_model(self):
        self.classifier_graph = tf.Graph()

        with self.classifier_graph.as_default():
            self.X = tf.placeholder(tf.int32, [None, self.max_padded_length])

            # Initial embedding initialized by custom word2vec vectors
            embedding_init = tf.Variable(tf.constant(self.word_vectors, shape=[self.vocab_size, self.embedding_dim]),
                                         name="word_embedding")

            self.embedding = tf.nn.embedding_lookup(embedding_init, self.X)

            # init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size])
            # self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)
            # self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)

            # enc_fw_cells = [tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)for layer in range(3)]
            # enc_bw_cells = [tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units) for layer in range(3)]

            # # Connect LSTM cells bidirectionally and stack
            # (all_states, fw_state, bw_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            #         cells_fw=enc_fw_cells, cells_bw=enc_bw_cells, inputs=self.embedding, dtype=tf.float32)
            # print('XXXXXX\n\n')
            # print(type(all_states))
            # print(tf.shape(all_states))
            # print(all_states)
            # # Concatenate results
            # for k in range(3):
            #     if k == 0:
            #         con_c = tf.concat((fw_state[k], bw_state[k]), 1)
            #         con_h = tf.concat((fw_state[k], bw_state[k]), 1)
            #     else:
            #         con_c = tf.concat((con_c, fw_state[k], bw_state[k]), 1)
            #         con_h = tf.concat((con_h, fw_state[k], bw_state[k]), 1)

            # output_2 = tf.contrib.rnn.LSTMStateTuple(c=con_c, h=con_h)
            # print("Stacked output shape: ",tf.shape(output_2))
            # rnn_op = tf.concat(output_2, 2)
            # print("Stacked rnn_op shape: ",tf.shape(rnn_op))
            # print(tf.shape(output_2))
            
            '''
            original
            forward_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)
            
            backward_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)

            output, state = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, self.embedding,
                                                            dtype=tf.float32)
            print(tf.shape(output))
            rnn_op = tf.concat(output, 2)
            '''
            
            forward_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)
            
            backward_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)

            output, state = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, self.embedding,
                                                            dtype=tf.float32)
            print("Regular Output shape: ",tf.shape(output))
            rnn_op_2 = tf.concat(output, 2)
            print("Regular rnn_op_shape: ",tf.shape(rnn_op_2))


            attn_layer_1 = tf.layers.dense(tf.reshape(rnn_op_2, (-1, 2 * self.no_lstm_units)),
                                           self.no_attn_hidden_128,
                                           activation=tf.nn.tanh, name="Attn1")
            print("Attn_layer_1: ",tf.shape(attn_layer_1))
            attn_extra = tf.layers.dense(attn_layer_1, self.no_attn_hidden_layer_1, activation=tf.nn.tanh, name="Attn_2_64")
            print("Attn_layer_extra: ",tf.shape(attn_extra))
            attn_layer_2 = tf.layers.dense(attn_extra, self.no_attn_hidden_layer_2, activation=tf.nn.tanh, name="Attn_2_32")
            print("Attn_layer_2: ",tf.shape(attn_layer_2))
            attention = tf.layers.dense(attn_layer_2, 1, activation=tf.nn.softmax, name="Attn2")
            print("Attn_2: ",tf.shape(attention))
            attention = tf.reshape(attention, (-1, self.max_padded_length, 1))
            print("Attention_reshape: ",tf.shape(attention))
            self.encoding_features = tf.squeeze(tf.matmul(tf.transpose(rnn_op_2, perm=[0, 2, 1]), attention), -1)
            print("Encoding features shape: ",tf.shape(self.encoding_features))

            print("Encoded Features: ", self.encoding_features)

            h_upp_0 = tf.layers.dense(self.encoding_features, self.upper_0, activation=tf.nn.relu, name="Classifier_hidden_upper_0")
            h_upp = tf.layers.dense(h_upp_0, self.upper, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001), name="Classifier_hidden_upper")
            # h_upp_2 = tf.layers.dense(self.encoding_features, self.upper_2, activation=tf.nn.relu, name="Classifier_hidden_upper_2")
            h1_0 = tf.layers.dense(h_upp, self.no_hidden_layer_1_0, activation=tf.nn.relu, name="Classifier_hidden1_0")
            h1_0_0 = tf.layers.dense(h1_0, self.no_hidden_layer_1_0_0, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001), name="Classifier_hidden1_0_0")
            dropout = tf.layers.dropout(h1_0_0,rate=0.3 )
            h1_1 = tf.layers.dense(dropout, self.no_hidden_layer_1_1, activation=tf.nn.relu, name="Classifier_hidden1_1")
            h1_1_0 = tf.layers.dense(h1_1, self.no_hidden_layer_1_1_0, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001) ,name="Classifier_hidden1_1_0")
            dropout_2 = tf.layers.dropout(h1_1_0,rate=0.3 )
            h1 = tf.layers.dense(dropout_2, self.no_hidden_layer_1, activation=tf.nn.relu, name="Classifier_hidden1") 
            h2 = tf.layers.dense(h1, self.no_hidden_layer_2, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001), name="Classifier_hidden2")
            dropout_3 = tf.layers.dropout(h2,rate=0.3 )
            print("Dropout 3 shape: ",tf.shape(dropout_3))
            self.logits = tf.layers.dense(dropout_3, self.no_class_labels, name="Classifier_logits")

            print(self.logits)
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.no_class_labels])
            self.d_loss = tf.placeholder(tf.float32, shape=[None,])

            self.classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))

            self.total_loss = self.classifier_loss + tf.losses.get_regularization_loss() + self.trade_off * self.d_loss

            # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            self.correct_indices = tf.argmax(self.y_, 1)
            self.predicted_indices = tf.argmax(tf.nn.softmax(self.logits), 1)
            self.correct_prediction = tf.equal(self.predicted_indices, self.correct_indices)

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            train_init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.classifier_sess = tf.Session(graph=self.classifier_graph)
        self.classifier_sess.run(train_init)

        self.discriminator_graph = tf.Graph()

        with self.discriminator_graph.as_default():
            self.discriminator_X = tf.placeholder(tf.float32, [None, 2 * self.no_lstm_units * self.no_class_labels],
                                                  name="DiscriminatorInp")
        
            d_h1_up_0 = tf.layers.dense(self.discriminator_X, self.no_disc_hidden_layer_upper_0, activation=tf.nn.relu,
                                   name="DiscriminatorH1_upper_0")
            d_h1_up = tf.layers.dense(d_h1_up_0 , self.no_disc_hidden_layer_upper, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                   name="DiscriminatorH1_upper")
            d_h1_0 = tf.layers.dense(d_h1_up, self.no_disc_hidden_layer_1_0, activation=tf.nn.relu,
                                   name="DiscriminatorH1_0")
            d_h1_middle = tf.layers.dense(d_h1_0, self.no_disc_hidden_layer_192, activation=tf.nn.relu,
                                   name="DiscriminatorH1_1_middle")
            d_h1_1 = tf.layers.dense(d_h1_middle, self.no_disc_hidden_layer_1_1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                   name="DiscriminatorH1_1")
            d_out = tf.layers.dropout(d_h1_1,rate=0.3 )
            d_h1_middle_96 = tf.layers.dense(d_out, self.no_disc_hidden_layer_96, activation=tf.nn.relu, 
                                   name="DiscriminatorH1_middle_96")
            d_h1 = tf.layers.dense(d_h1_middle_96, self.no_disc_hidden_layer_1, activation=tf.nn.relu, 
                                   name="DiscriminatorH1")
            d_h2 = tf.layers.dense(d_h1, self.no_disc_hidden_layer_2, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001), name="DiscriminatorH2")
            d_out_2 = tf.layers.dropout(d_h2,rate=0.3)
            d_op = tf.layers.dense(d_out_2, 1, activation=tf.nn.sigmoid, name="DiscriminatorOutput")
            print("d_op shape: ",tf.shape(d_op))
            self.discriminator_y = tf.placeholder(tf.float32, [None, 1], name="DiscriminatorOup")

            self.discriminator_loss = tf.losses.sigmoid_cross_entropy(self.discriminator_y, logits=d_op)
            self.discriminator_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.discriminator_loss)

            train_init = tf.global_variables_initializer()
            self.discriminator_saver = tf.train.Saver()

        self.discriminator_sess = tf.Session(graph=self.discriminator_graph)
        self.discriminator_sess.run(train_init)

    def get_batches(self, X, Y, bsize):
        for i in range(0, len(X) - bsize + 1, bsize):
            indices = slice(i, i + bsize)
            yield X[indices], Y[indices]

    def train(self):
        print("Training...")

        log = open(self.result_dir + 'log', 'w')
        log.write('#Index, Training Time, Training Loss, Validation Loss, Train Acc., Validation Acc.\n')
        log.close()

        highest_val_acc = 0

        target_batch_index = 0
        target_data_length = len(self.target_train)

        no_batch_iter = int(target_data_length / self.batch_size)
        for i in range(self.iterations):
            print(("Iteration: " + str(i + 1)))
            start = time.time()
            training_error = 0
            t_count = 0
            t_acc = 0

            for j in range(no_batch_iter):
                target_indices = slice(target_batch_index, target_batch_index + self.batch_size)

                target_batch_xs,target_batch_ys = self.target_train[target_indices], self.target_train_label[target_indices]
                # print("XXXXXXXXXXXX: \n")
                # print(target_batch_xs)
                feed_dict = {self.X: target_batch_xs, self.y_: target_batch_ys, self.d_loss: np.array([0]*42)}
                t_e, _, t_a = self.classifier_sess.run(
                    [self.classifier_loss, self.train_step, self.accuracy], feed_dict=feed_dict)

                t_acc += t_a
                training_error += t_e
                t_count += 1

                target_batch_index = (target_batch_index+self.batch_size)%target_data_length

            training_time = str(time.time() - start)
            training_error /= t_count

            t_acc /= t_count

            # Apply Learning rate decay for SGD.
            # learning_rate = learning_rate * decay_factor

            print(("Time : " + str(training_time) + " Seconds" + " training_error : " + str(training_error)))
            print("Train acc : ", t_acc * 100)
            log = open(self.result_dir + 'log', 'a')
            log.write(str(i + 1) + ' ' + training_time + ' ' + str(training_error) + ' ' + str(t_acc * 100) + '\n')
            log.close()

            if t_acc > highest_val_acc:
                highest_val_acc = t_acc
                save_path = self.saver.save(self.classifier_sess, self.path_to_model)
                print(("Model saved in file : %s" % save_path))

        print("Done Training :)")

    def test(self, load_saved_model=True):
        print("STARTING TESTING")
        test_acc = 0
        if load_saved_model:
            self.saver.restore(self.classifier_sess, self.path_to_model)

            feed_dict = {self.X: self.target_test, self.y_: self.target_test_label}

            correct_indices, predicted_indices, test_acc = self.classifier_sess.run(
                [self.correct_indices, self.predicted_indices, self.accuracy],
                feed_dict=feed_dict)

            import collections
            class_wise_example_count = collections.Counter(correct_indices)
            class_wise_correct_predicted = {}
            for i in range(len(correct_indices)):
                if correct_indices[i] == predicted_indices[i]:
                    if correct_indices[i] in class_wise_correct_predicted.keys():
                        class_wise_correct_predicted[correct_indices[i]] += 1
                    else:
                        class_wise_correct_predicted[correct_indices[i]] = 1

        print("Test Accuracy: ", test_acc)
        with open("./save_test_2_update/after_train_accuracy.txt", "a") as f:
            f.write("Test Accuracy = " + str(test_acc) + "\n")
            f.write("Class,Count,Correct Predicted\n")

            for key in class_wise_example_count.keys():
                if key in class_wise_correct_predicted.keys():
                    f.write(str(key) + "," + str(class_wise_example_count[key]) + "," + str(
                        class_wise_correct_predicted[key]) + "\n")
                else:
                    f.write(str(key) + "," + str(class_wise_example_count[key]) + ",0\n")


if __name__ == '__main__':
    model = CDAN()
    model.train()
    model.test()
