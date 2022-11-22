import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine as dist
from sklearn.metrics import confusion_matrix

# Added to filter out 'stderr' print statements from 'confusion_matrix(y_true, y_pred)'
#   using: '2> /dev/null'
import sys

class StarSpaceShip:

    def __init__(self):
      ''' '''
      self.input_encoder_model = None
      self.target_encoder_model = None
      self.model = None

      self.target_encodings = None
      self.distance_dict = None

      self.test_positive_input_batches = None
      self.test_positive_batch_targets = None
      self.test_negative_batch_targets = None
      
    # Called from 'prepare_features_targets()'
    @staticmethod
    def _fetch_prepare_rawdata():

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return x_train, y_train, x_test, y_test

    # Called from 'build_model_architecture()'
    @staticmethod
    def custom_loss(y_true, y_pred):

        pos_input_emb = y_pred[:, 0, :]
        pos_target_emb = y_pred[:, 1, :]
        neg_target_embs = y_pred[:, 2:, :]

        scores_pos = tf.expand_dims(tf.reduce_sum(tf.multiply(pos_input_emb, pos_target_emb), -1), axis = -1)
        scores_neg = tf.expand_dims(tf.reduce_sum(tf.multiply(pos_input_emb, tf.math.reduce_mean(neg_target_embs, axis = 1)), -1), axis = -1)

        loss_matrix = tf.maximum(0., 1. - scores_pos + scores_neg)
        loss = tf.reduce_sum(loss_matrix)

        return loss

    # Called from 'train_star_space()'
    def prepare_features_targets(self):
        x_train, y_train, x_test, y_test = self._fetch_prepare_rawdata()

        train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets = self.generate_batches(x_train, y_train)
        test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets = self.generate_batches(x_test, y_test)

        train_dummy_outputs = np.zeros((len(train_positive_input_batches), 12, 256))
        test_dummy_outputs = np.zeros((len(test_positive_input_batches), 12, 256))

        train_positive_input_batches = train_positive_input_batches.reshape(len(train_positive_input_batches), 1, 784)
        train_positive_batch_targets = train_positive_batch_targets.reshape(len(train_positive_batch_targets), 1)
        train_negative_batch_targets = train_negative_batch_targets.reshape(len(train_negative_batch_targets), 10)

        test_positive_input_batches = test_positive_input_batches.reshape(len(test_positive_input_batches), 1, 784)
        test_positive_batch_targets = test_positive_batch_targets.reshape(len(test_positive_batch_targets), 1)
        test_negative_batch_targets = test_negative_batch_targets.reshape(len(test_negative_batch_targets), 10)

        return train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets, train_dummy_outputs,\
             test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets, test_dummy_outputs

    # Called from 'prepare_features_targets()'
    def generate_batches(self, x, y):
        positive_input_batches = []
        negative_input_batches = []
        positive_batch_targets = []
        negative_batch_targets = []

        for idx, x_feats in enumerate(x):

            positive_input_batch = [x[idx]]
            positive_batch_target = [y[idx]]
            negative_batch_target = []

            for neg_idx in random.sample(list(np.where(y[:, 0] != y[idx][0])[0]), 10):
                negative_batch_target.append(y[neg_idx])

            positive_input_batches.append(np.array(positive_input_batch))
            positive_batch_targets.append(np.array(positive_batch_target))
            negative_batch_targets.append(np.array(negative_batch_target))

        return np.array(positive_input_batches), np.array(positive_batch_targets), np.array(negative_batch_targets)

    # Called from 'train_star_space()'
    def build_model_architecture(self):
        positive_input = tf.keras.layers.Input(shape = (1, 784))
        positive_target_input = tf.keras.layers.Input(shape = (1, ))
        negative_target_inputs = tf.keras.layers.Input(shape = (10, ))

        input_dense_layer = tf.keras.layers.Dense(256)
        target_embedding_layer = tf.keras.layers.Embedding(input_dim = 10, output_dim = 256)
        target_dense_layer = tf.keras.layers.Dense(256)

        pos_input_embedding = tf.nn.l2_normalize(input_dense_layer(positive_input), -1)

        pos_target_embedding = tf.nn.l2_normalize(target_dense_layer(target_embedding_layer(positive_target_input)), -1)
        neg_target_embedding = tf.nn.l2_normalize(target_dense_layer(target_embedding_layer(negative_target_inputs)), -1)

        packed_output_embeddings = tf.keras.layers.concatenate([pos_input_embedding, pos_target_embedding, neg_target_embedding], axis = 1)

        self.model = tf.keras.models.Model(inputs = [positive_input, positive_target_input, negative_target_inputs], outputs = packed_output_embeddings)

        self.model.compile(loss = self.custom_loss, optimizer = 'adam')

        self.input_encoder_model = tf.keras.models.Model(inputs = positive_input, outputs = pos_input_embedding)
        self.target_encoder_model = tf.keras.models.Model(inputs = positive_target_input, outputs = pos_target_embedding)

    # ================================= PUBLIC METHODS =============================================
    
    # Added to direct print statements to 'stderr' to filter out TensorFlow 'stdout' msgs 
    #   using: '2>&1 > /dev/null'

    # 'model.summary(print_fn=XXX)' can handle either 'staticmethod' or object call with 'self'
    #@staticmethod
    def print2StdErr(self, str):
      print(str, file=sys.stderr)

    def train_star_space(self):
        train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets, train_dummy_outputs,\
             test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets, test_dummy_outputs = self.prepare_features_targets()

        self.test_positive_input_batches = test_positive_input_batches
        self.test_positive_batch_targets = test_positive_batch_targets
        self.test_negative_batch_targets = test_negative_batch_targets

        self.build_model_architecture()
        
        
        print("\nModel summary")
        print("-------------")
        # If 'self.model.summary()' is printed then the return value is printed ie "None"
        #   https://keras.io/api/models/model/#summary-method - "Defaults to print"
        self.model.summary()
        #self.model.summary(print_fn=self.print2StdErr)
        #print (self.model.summary())
        print("=============")

        print("\nTraining model...")
        self.model.fit([train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets], train_dummy_outputs, epochs = 10,\
         validation_data = ([test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets], test_dummy_outputs))

        print("\nPredicting target encodings...")
        self.target_encodings = {target_id: self.target_encoder_model.predict(np.array([target_id]))[0, 0, :] for target_id in range(10)}

        d = {}
        for i in range(10):
            for j in range(10):
                if i != j and f'{j}_{i}' not in d:
                    d[f'{i}_{j}'] = 1 - dist(self.target_encodings[i], self.target_encodings[j])

        print("\nTarget encodings")
        print("----------------")
        print ({k: v for k, v in sorted(d.items(), key=lambda item: item[1])})

    # Called from 'plot_confusion_matrix()'
    def predict_class(self, input_image = None):

        actual_target = None
        if input_image is None and self.test_positive_input_batches is not None:
            random_idx = np.random.randint(0, len(self.test_positive_input_batches))
            input_image = self.test_positive_input_batches[random_idx, :].reshape(1, 1, 784)
            actual_target = self.test_positive_batch_targets[random_idx]

        if self.input_encoder_model:
            input_encodings = self.input_encoder_model.predict(input_image)[0, 0, :]

        distance_dict = {i: 1.0 - dist(self.target_encodings[i], input_encodings) for i in range(10)}
        
        return distance_dict, actual_target

    def save_projector_tensorflow_files(self):
        '''
        This method prepares the held-out set to enable PCA visualization using https://projector.tensorflow.org/
        To do this we need two TSV files one containing the floating point tab serperated embeddings file
        The other file has labels for each of the rows
        '''
        testset_embeddings = self.input_encoder_model.predict(self.test_positive_input_batches)
        testset_embeddings = testset_embeddings[:, 0, :].astype('U25')
        
        #with open("visualization/projector_tensorflow_data/test_embedding_vectors.tsv", "w") as f:
        with open("visualization/test_embedding_vectors.tsv", "w") as f:
            f.write("\n".join(["\t".join(testset_embedding) for testset_embedding in testset_embeddings]))

        f.close()

        #with open("visualization/projector_tensorflow_data/test_embedding_labels.tsv", "w") as f:
        with open("visualization/test_embedding_labels.tsv", "w") as f:
            f.write("\n".join([str(label[0]) for label in self.test_positive_batch_targets]))

        f.close()

    def plot_confusion_matrix(self):
        labels = list(range(10))
        y_pred = []
        y_true = []
        for idx, test_positive_input in enumerate(self.test_positive_input_batches):
            scores = [score for _, score in self.predict_class(test_positive_input.reshape(1, 1, 784))[0].items()]
            y_pred.append(np.argmax(scores))
            y_true.append(self.test_positive_batch_targets[idx][0])

        # "TypeError: confusion_matrix() takes 2 positional arguments but 3 were given"
        #cm = confusion_matrix(y_true, y_pred, labels)
        # 'scikit-learn' v0.n and v1.1 had diff positional vs keyword arg set

        cm = confusion_matrix(y_true, y_pred)
        
        labels = [str(label) for label in labels]

        plt.imshow(cm, interpolation='nearest')
        plt.xticks(np.arange(0, 10), labels)
        plt.yticks(np.arange(0, 10), labels)

        plt.savefig("visualization/confusion_matrix.png")

if __name__ == "__main__":
  print("\nHello world")

  nn = StarSpaceShip()
  
  print("\nCalling 'StarSpaceShip::train_star_space()'")
  nn.train_star_space()

  # https://discuss.python.org/t/builtin-function-input-writes-its-prompt-to-sys-stderr-and-not-to-sys-stdout/12955/4
  question = "\nDo you want to run 'plot_confusion_matrix()', which takes about 10mins (Y/n)?"
  print (question,end='') # ie no newline since 'input()' prints to 'stderr' which is getting filtered out
  
  # Not cause 'input()' to print to 'stdout' (so must be getting redirected in input() stack)
  #   (like can be done the other way for 'plot_confusion_matrix()')
  #sys.stderr = sys.stdout
  #response = input(question)

  response = input()
  response = response.lower()
  if (response and response[0] == 'y') or response == '':
    print("\nCalling 'StarSpaceShip::plot_confusion_matrix()'")

    # 'confusion_matrix(y_true, y_pred)' prints a lot of progress lines to 'stdout' so removed 
    #   with '2> /dev/null' on cmd line to filter out 'stdout' (with NO SPACE between '2' and '>')
    sys.stdout = sys.stderr
    nn.plot_confusion_matrix()
    sys.stdout = sys.__stdout__
  elif (response and response[0] == 'n'):
    print('no problem then')
  else:
    print('default is Yes but should be caught earlier')
  
  question = "\nDo you want to run 'save_projector_tensorflow_files()'(Y/n)?"
  print (question,end='')

  response = input()
  response = response.lower()
  if (response and response[0] == 'y') or response == '':
    print("\nCalling 'StarSpaceShip::save_projector_tensorflow_files()'")
    nn.save_projector_tensorflow_files()

