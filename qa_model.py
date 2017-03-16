from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

from tensorflow.python.ops.nn import birectional_dynamic_rnn
from tensorflow.contrib import rnn

from util import Progbar, minibatches

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim, qlen, c_len):
        self.size = size
        self.vocab_dim = vocab_dim
        self.max_question_length = q_len
        self.max_context_length = c_len

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        questions, contexts = inputs
        q_mask, c_mask = masks

        cell_fw = rnn.LSTMCell(self.size)
        cell_bw = rnn.LSTMCell(self.size)
        q_outputs, q_output_states = bidirectional_dynamic_rnn(cell_fw,
                                                               cell_bw, 
                                                               questions,
                                                               sequence_length=q_mask, 
                                                               # initial_state_fw=encoder_state_input, 
                                                               # initial_state_bw=encoder_state_input,
                                                               time_major=True,
                                                               dtype=dtypes.float32)

        

        # question_rep = tf.concat(output_states, 0)

        c_outputs, c_output_states = bidirectional_dynamic_rnn(cell_fw,
                                                               cell_bw, 
                                                               contexts,
                                                               sequence_length=c_mask, 
                                                               initial_state_fw=q_output_states[0], 
                                                               initial_state_bw=q_output_states[1],
                                                               time_major=True,
                                                               dtype=dtypes.float32)
        
        return c_output_states

    # TODO
    
    
    

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        dropout_rate = self.dropout_placeholder
        ### YOUR CODE HERE (~10-20 lines)
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('decoder') as scope:
            W = tf.get_variable(name="W", shape=(2*self.state_size, self.config.hidden_size), dtype=tf.float32, initializer=xavier_initializer)
            b1 = tf.Variable(tf.zeros((self.config.hidden_size,)), name="b1")
            U = tf.get_variable(name="U", shape=(self.config.hidden_size, self.config.max_context_length), dtype=tf.float32, initializer=xavier_initializer)
            b2 = tf.Variable(tf.zeros((self.config.max_context_length)), name="b2")

        h = tf.nn.relu(tf.matmul(x, W) + b1)
        h_drop = tf.nn.dropout(h, self.dropout_placeholder)
        pred = tf.matmul(h_drop, U) + b2

        #pred = tf.layers.dense(knowledge_rep, )
        return pred               


class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.add_placeholders()


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(embeddings)
            self.setup_system()
            self.setup_loss()

            optimizer = get_optimizer("adam")
            self.train_op = optimizer(FLAGS.learning_rate).minimize(loss)


        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoding = encoder.encode((self.question_input_placeholder, self.context_input_placeholder), 
                                  (self.question_masks_placeholder, self.context_masks_placeholder))

        self.pred_start = decoder.decode(encoding)
        self.pred_end = decoder.decode(encoding)



    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            modified_pred_start = self.pred_start + tf.log(tf.one_hot(self.context_masks_placeholder, self.max_context_length))
            modified_pred_end = self.pred_end + tf.log(tf.one_hot(self.context_masks_placeholder, self.max_context_length))

            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[0])
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[1])

            loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)

    # TODO
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        # with vs.variable_scope("embeddings"):
        #     L = tf.Variable(self.pretrained_embeddings, name="L")

        with tf.variable_scope('model') as scope:
            L = tf.Variable(self.pretrained_embeddings, name="L")
        
        # print L.get_shape()
        # print self.input_placeholder.get_shape()

        # embeddings = tf.reshape(tf.nn.embedding_lookup(L, self.input_placeholder), [-1, self.max_length, Config.n_features * Config.embed_size])

    def add_placeholders(self): 
        self.question_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_question_length), name="questions")
        self.question_masks_placeholder = tf.placeholder(tf.int32, shape=(), name="question_masks")
        # self.question_masks_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_question_length), name="question_masks")

        self.context_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_context_length), name="contexts")
        self.context_masks_placeholder = tf.placeholder(tf.int32, shape=(), name="context_masks")

        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, FLAGS.labels_size), name="labels")

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")

    # TODO
    def create_feed_dict(self, q_inputs_batch=None, q_mask_batch=None, c_inputs_batch=None, c_mask_batch=None, labels_batch=None, dropout=1):
        feed_dict = {}
        if q_inputs_batch != None: 
            feed_dict[self.question_input_placeholder] = q_inputs_batch
        if q_mask_batch != None: 
            feed_dict[self.question_masks_placeholder] = q_mask_batch
        if c_inputs_batch != None: 
            feed_dict[self.context_input_placeholder] = c_inputs_batch
        if c_mask_batch != None: 
            feed_dict[self.context_masks_placeholder] = c_mask_batch
        if labels_batch != None: 
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout != None: 
            feed_dict[self.dropout_placeholder] = dropout
        
        return feed_dict

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / FLAGS.batch_size))
        losses, grad_norms = [], []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss, grad_norm = self.train_on_batch(sess, *batch)
            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])

        return losses, grad_norms

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        losses, grad_norms = [], []
        for epoch in range(FLAGS.epochs):
            logger.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
            loss, grad_norm = self.run_epoch(session, dataset)

            losses.append(loss)
            grad_norms.append(grad_norm)