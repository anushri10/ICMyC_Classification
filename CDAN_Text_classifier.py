import tensorflow as tf
import numpy as np
import time
from numpy import linalg as la
import pickle


class CDAN:
    def __init__(self):
        self.result_dir = "./save/"
        self.model_name = "model.ckpt"
        self.path_to_model = self.result_dir + self.model_name
        self.batch_size = 100
        self.iterations = 100

        self.learning_rate = 1e-3
        self.embedding_dim = 100
        # self.no_class_labels = None
        self.trade_off = 0.3
        self.no_attn_hidden_layer_2 = 32
        self.no_attn_hidden_layer_1 = 64
        self.no_hidden_layer_1 = 64
        self.no_hidden_layer_2 = 32
        self.no_disc_hidden_layer_1 = 64
        self.no_disc_hidden_layer_2 = 32
        self.max_padded_length = 30
        self.no_lstm_units = 128

        self.load_data()
        self.define_model()

    def load_data(self):
        file_path = r"E:\Study\M.Tech\R&D2\Work\TransferLearning\CDAN\AmazonYelpReview"

        # load pre-trained word embedding.
        with open(file_path + "/word_vector.pkl", "rb") as myFile:
            self.word_vectors = pickle.load(myFile, encoding='latin1')

        self.word_vectors = np.asarray(self.word_vectors).astype(np.float32)

        for i in range(len(self.word_vectors) - 1):
            self.word_vectors[i] /= (la.norm((self.word_vectors[i])))

        # load target train sentences and labels
        with open(file_path + "/target_train_labels.pkl", "rb") as myFile:
            # OF NO USE.. as this is unsupervised domain adaptation
            self.target_train_label = pickle.load(myFile, encoding='latin1')
        with open(file_path + "/target_train_sentence_word_index.pkl", "rb") as myFile:
            self.target_train = pickle.load(myFile, encoding='latin1')

        # load target test sentences and labels
        with open(file_path + "/target_test_labels.pkl", "rb") as myFile:
            self.target_test_label = pickle.load(myFile, encoding='latin1')
        with open(file_path + "/target_test_sentence_word_index.pkl", "rb") as myFile:
            self.target_test = pickle.load(myFile, encoding='latin1')

        # load source sentences and labels
        with open(file_path + "/source_train_labels.pkl", "rb") as myFile:
            self.source_label = pickle.load(myFile, encoding='latin1')
        with open(file_path + "/source_train_sentence_word_index.pkl", "rb") as myFile:
            self.source_train = pickle.load(myFile, encoding='latin1')

        # reduce 16 lakh review text to 1 lakh for fast training
        index = np.random.choice(self.source_train.shape[0],100000,replace=True)
        self.source_train = self.source_train[index]
        self.source_label = self.source_label[index]

        self.vocab_size = len(self.word_vectors)
        self.no_class_labels = len(self.source_label[0])

        print("Source Sentence Count: ", len(self.source_train))
        print("Target Sentence Count: ", len(self.target_train))
        print("Target Test Sentence Count: ", len(self.target_test))

    def define_model(self):
        self.classifier_graph = tf.Graph()

        with self.classifier_graph.as_default():
            self.X = tf.placeholder(tf.int32, [None, self.max_padded_length])

            # Initial embedding initialized by word2vec vectors
            embedding_init = tf.Variable(tf.constant(self.word_vectors, shape=[self.vocab_size, self.embedding_dim]),
                                         name="word_embedding")

            self.embedding = tf.nn.embedding_lookup(embedding_init, self.X)

            forward_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)
            backward_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.no_lstm_units)

            output, state = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, self.embedding,
                                                            dtype=tf.float32)
            rnn_op = tf.concat(output, 2)

            attn_layer_1 = tf.layers.dense(tf.reshape(rnn_op, (-1, 2 * self.no_lstm_units)),
                                           self.no_attn_hidden_layer_1,
                                           activation=tf.nn.tanh, name="Attn1")
            attention = tf.layers.dense(attn_layer_1, 1, activation=tf.nn.softmax, name="Attn2")
            attention = tf.reshape(attention, (-1, self.max_padded_length, 1))
            self.encoding_features = tf.squeeze(tf.matmul(tf.transpose(rnn_op, perm=[0, 2, 1]), attention), -1)

            print("Encoded Features: ", self.encoding_features)

            h1 = tf.layers.dense(self.encoding_features, self.no_hidden_layer_1, activation=tf.nn.relu,
                                 name="Classifier_hidden1")
            h2 = tf.layers.dense(h1, self.no_hidden_layer_2, activation=tf.nn.relu, name="Classifier_hidden2")
            self.logits = tf.layers.dense(h2, self.no_class_labels, name="Classifier_logits")

            print(self.logits)
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.no_class_labels])
            self.d_loss = tf.placeholder(tf.float32, shape=[None, 1])

            self.classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))

            self.total_loss = self.classifier_loss + self.trade_off * self.d_loss

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
            d_h1 = tf.layers.dense(self.discriminator_X, self.no_disc_hidden_layer_1, activation=tf.nn.relu,
                                   name="DiscriminatorH1")
            d_h2 = tf.layers.dense(d_h1, self.no_disc_hidden_layer_2, activation=tf.nn.relu, name="DiscriminatorH2")
            d_op = tf.layers.dense(d_h2, 1, activation=tf.nn.sigmoid, name="DiscriminatorOutput")
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

                feed_dict = {self.X: target_batch_xs, self.y_: target_batch_ys, self.d_loss: np.array([0]*len(source_batch_ys))}
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
        with open("after_train_accuracy.txt", "a") as f:
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
