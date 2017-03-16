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

# from tensorflow.python.ops.nn import birectional_dynamic_rnn
# from tensorflow.contrib import rnn

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
    def __init__(self, size, vocab_dim, config):
        self.size = size
        self.vocab_dim = vocab_dim
        self.max_question_length = config.max_question_length
        self.max_context_length = config.max_context_length
        self.config = config

    def encode(self, inputs, masks, encoder_state_input=None):
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

        print(questions)
        print(contexts)
        print(q_mask)
        print(c_mask)

        with tf.variable_scope('q_decoder') as scope:
            q_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.state_size, state_is_tuple=True)
            q_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.state_size, state_is_tuple=True)
            q_outputs, q_output_states = tf.nn.bidirectional_dynamic_rnn(q_cell_fw,
                                                                       q_cell_bw, 
                                                                       questions,
                                                                       sequence_length=q_mask, 
                                                                       # initial_state_fw=encoder_state_input, 
                                                                       # initial_state_bw=encoder_state_input,
                                                                       time_major=True,
                                                                       dtype=tf.float32)
        print(q_output_states[0])

        with tf.variable_scope('p_decoder') as scope:
            # question_rep = tf.concat(output_states, 0)
            p_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.state_size, state_is_tuple=True)
            p_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.state_size, state_is_tuple=True)

            
            c_outputs, c_output_states = tf.nn.bidirectional_dynamic_rnn(p_cell_fw,
                                                                       p_cell_bw, 
                                                                       contexts,
                                                                       sequence_length=c_mask, 
                                                                       initial_state_fw=q_output_states[0], 
                                                                       initial_state_bw=q_output_states[1],
                                                                       time_major=True,
                                                                       dtype=tf.float32)
        # print(c_output_states[0].c)
        #states = tf.unpack(c_output_states[0])
        #print(states)
        #final_rep = tf.concat(0, states)
        final_rep = tf.concat(1, (c_output_states[0].h, c_output_states[1].h))
        print(c_output_states[0].h)
        print(final_rep)
        logging.info("Done adding encode definition")
        return final_rep

    # TODO
    
    
    

class Decoder(object):
    def __init__(self, output_size, config):
        self.output_size = output_size
        self.config = config

    def decode(self, knowledge_rep, dropout_rate, isTraining):
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

        ## YOUR CODE HERE (~10-20 lines)
        # xavier_initializer = tf.contrib.layers.xavier_initializer()
        # with tf.variable_scope('decoder') as scope:
        #     dense = tf.contrib.layers.relu(inputs=knowledge_rep)#, units=1024)#, activation=tf.nn.relu)
        # pred = tf.contrib.layers.dropout(inputs=dense, rate=dropout_rate, training=isTraining)

        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('decoder') as scope:
            W = tf.get_variable(name="W", shape=(2 * self.config.state_size, self.config.hidden_size), dtype=tf.float32, initializer=xavier_initializer) #
            b1 = tf.Variable(tf.zeros((self.config.hidden_size,)), name="b1")
            U = tf.get_variable(name="U", shape=(self.config.hidden_size, self.config.max_context_length), dtype=tf.float32, initializer=xavier_initializer)
            b2 = tf.Variable(tf.zeros((self.config.max_context_length)), name="b2")
        
        print(knowledge_rep) #.shape, W.shape, U.shape

        h = tf.nn.relu(tf.batch_matmul(knowledge_rep, W) + b1)
        h_drop = tf.nn.dropout(h, dropout_rate)
        pred = tf.matmul(h_drop, U) + b2

        #pred = tf.layers.dense(knowledge_rep, )
        logging.info("Done adding decode definition")
        return pred  

    def decode_start(self, knowledge_rep, dropout_rate, isTraining):
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

        ## YOUR CODE HERE (~10-20 lines)
        # xavier_initializer = tf.contrib.layers.xavier_initializer()
        # with tf.variable_scope('decoder') as scope:
        #     dense = tf.contrib.layers.relu(inputs=knowledge_rep)#, units=1024)#, activation=tf.nn.relu)
        # pred = tf.contrib.layers.dropout(inputs=dense, rate=dropout_rate, training=isTraining)

        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('decoder_start') as scope:
            W = tf.get_variable(name="W", shape=(2 * self.config.state_size, self.config.hidden_size), dtype=tf.float32, initializer=xavier_initializer) #
            b1 = tf.Variable(tf.zeros((self.config.hidden_size,)), name="b1")
            U = tf.get_variable(name="U", shape=(self.config.hidden_size, self.config.max_context_length), dtype=tf.float32, initializer=xavier_initializer)
            b2 = tf.Variable(tf.zeros((self.config.max_context_length)), name="b2")
        
        print(knowledge_rep) #.shape, W.shape, U.shape

        h = tf.nn.relu(tf.batch_matmul(knowledge_rep, W) + b1)
        h_drop = tf.nn.dropout(h, dropout_rate)
        pred = tf.matmul(h_drop, U) + b2

        #pred = tf.layers.dense(knowledge_rep, )
        logging.info("Done adding decode definition")
        return pred  

    def decode_end(self, knowledge_rep, dropout_rate, isTraining):
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

        ## YOUR CODE HERE (~10-20 lines)
        # xavier_initializer = tf.contrib.layers.xavier_initializer()
        # with tf.variable_scope('decoder') as scope:
        #     dense = tf.contrib.layers.relu(inputs=knowledge_rep)#, units=1024)#, activation=tf.nn.relu)
        # pred = tf.contrib.layers.dropout(inputs=dense, rate=dropout_rate, training=isTraining)

        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('decoder_end') as scope:
            W = tf.get_variable(name="W", shape=(2 * self.config.state_size, self.config.hidden_size), dtype=tf.float32, initializer=xavier_initializer) #
            b1 = tf.Variable(tf.zeros((self.config.hidden_size,)), name="b1")
            U = tf.get_variable(name="U", shape=(self.config.hidden_size, self.config.max_context_length), dtype=tf.float32, initializer=xavier_initializer)
            b2 = tf.Variable(tf.zeros((self.config.max_context_length)), name="b2")
        
        print(knowledge_rep) #.shape, W.shape, U.shape

        h = tf.nn.relu(tf.batch_matmul(knowledge_rep, W) + b1)
        h_drop = tf.nn.dropout(h, dropout_rate)
        pred = tf.matmul(h_drop, U) + b2

        #pred = tf.layers.dense(knowledge_rep, )
        logging.info("Done adding decode definition")
        return pred              


class QASystem(object):
    def __init__(self, encoder, decoder, embeddings, config,  *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.config = config
        self.pretrained_embeddings = embeddings
        self.encoder = encoder
        self.decoder = decoder

        # ==== set up placeholder tokens ========
        self.add_placeholders()


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

            optimizer = get_optimizer("adam")
            self.train_op = optimizer(self.config.learning_rate).minimize(self.loss)

        logging.info("Done setup!!!!!!!!!!!")


        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoding = self.encoder.encode((self.question_input_placeholder, self.context_input_placeholder), 
                                       (self.question_masks_placeholder, self.context_masks_placeholder))

        self.pred_start = self.decoder.decode_start(encoding, self.dropout_placeholder, True)
        self.pred_end = self.decoder.decode_end(encoding, self.dropout_placeholder, True)



    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            modified_pred_start = self.pred_start + tf.log(tf.one_hot(self.context_masks_placeholder, self.config.max_context_length))
            modified_pred_end = self.pred_end + tf.log(tf.one_hot(self.context_masks_placeholder, self.config.max_context_length))

            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[0])
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[1])

            self.loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)

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

        logging.info("Done adding embeddings")
        
        # print L.get_shape()
        # print self.input_placeholder.get_shape()

        # embeddings = tf.reshape(tf.nn.embedding_lookup(L, self.input_placeholder), [-1, self.max_length, Config.n_features * Config.embed_size])

    def add_placeholders(self): 
        self.question_input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_question_length, 1), name="questions")
        self.question_masks_placeholder = tf.placeholder(tf.int32, shape=(None,), name="question_masks")
        # self.question_masks_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_question_length), name="question_masks")

        self.context_input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_context_length, 1), name="contexts")
        self.context_masks_placeholder = tf.placeholder(tf.int32, shape=(None,), name="context_masks")

        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.label_size), name="labels")

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")

        logging.info("Done adding placeholders")

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

        batch_size = len(train_y)
        # input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        # input_feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        q_inputs_batch, q_mask_batch = zip(*(train_x[0])) 
        c_inputs_batch, c_mask_batch = zip(*(train_x[1])) 

        c_inputs_batch = np.array(c_inputs_batch).reshape(batch_size, self.config.max_context_length, 1)
        q_inputs_batch = np.array(q_inputs_batch).reshape(batch_size, self.config.max_question_length, 1)
        
        input_feed = self.create_feed_dict(q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch, train_y, self.config.dropout)

        # output_feed = []

        # outputs = session.r
        _, loss = session.run([self.train_op, self.loss], feed_dict=input_feed)
        return loss

        # output_feed = []

        # outputs = session.run(output_feed, input_feed)

        # return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        # input_feed = {}

        # # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        # q_inputs_batch, q_mask_batch = zip(*(valid_x[0])) 
        # c_inputs_batch, c_mask_batch = zip(*(valid_x[1])) 

        # input_feed = self.create_feed_dict(q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch, valid_y, self.config.dropout)

        # output_feed = []

        # outputs = session.run(output_feed, input_feed)

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

    def get_batch_from_indices(self, data, indices):
        return [data[i] for i in indices]

    def get_minibatches(self, indices, minibatch_size, shuffle=True):
        data_size = len(indices)
        if shuffle:
            np.random.shuffle(indices)
        batch_indices = []
        for minibatch_start in np.arange(0, data_size, minibatch_size):
            batch_indices.append(indices[minibatch_start:minibatch_start + minibatch_size]) 
        return batch_indices

    def run_epoch(self, sess, train):
        train_indices = range(len(train['q']))
        prog = Progbar(target=1 + int(len(train['q']) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch_indices in enumerate(self.get_minibatches(train_indices, self.config.batch_size)):
            q_batch = self.get_batch_from_indices(train['q'], batch_indices)
            c_batch = self.get_batch_from_indices(train['c'], batch_indices)
            s_batch = self.get_batch_from_indices(train['s'], batch_indices)


            loss = self.optimize(sess, (q_batch, c_batch), s_batch)
            # losses.append(loss)
            # grad_norms.append(grad_norm)
            # loss = 0
            print(self.get_batch_from_indices(train['s'], batch_indices))
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

        self.train_data = dataset[0]
        self.test_data = dataset[1]

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        losses, grad_norms = [], []
        for epoch in range(self.config.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            loss, grad_norm = self.run_epoch(session, self.train_data)

            losses.append(loss)
            grad_norms.append(grad_norm)