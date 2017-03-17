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

from random import sample as random_sample

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
                                                                       swap_memory=True,
                                                                       time_major=False,
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
                                                                       swap_memory=True,
                                                                       time_major=False,
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
    def __init__(self, encoder, decoder, embeddings, config, vocab,  *args):
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
        self.vocab = vocab[0]
        self.rev_vocab = vocab[1]

        # ==== set up placeholder tokens ========
        self.add_placeholders()


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

            optimizer = get_optimizer("adam")
            self.train_op = optimizer(self.config.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()

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
            # print('SHAPE', self.context_masks_placeholder)
            epsilon = tf.constant(value=0.0000001)

            modified_pred_start = self.pred_start + tf.log(epsilon + tf.one_hot(indices=self.context_masks_placeholder, depth=self.config.max_context_length))
            modified_pred_end = self.pred_end + tf.log(epsilon + tf.one_hot(indices=self.context_masks_placeholder, depth=self.config.max_context_length))

            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[:,0])
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[:,1])

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
        # print q_mask_batch
        if q_inputs_batch is not None: 
            feed_dict[self.question_input_placeholder] = q_inputs_batch
        if q_mask_batch is not None: 
            feed_dict[self.question_masks_placeholder] = q_mask_batch
        if c_inputs_batch is not None: 
            feed_dict[self.context_input_placeholder] = c_inputs_batch
        if c_mask_batch is not None: 
            feed_dict[self.context_masks_placeholder] = c_mask_batch
        if labels_batch is not None: 
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None: 
            feed_dict[self.dropout_placeholder] = dropout
        
        return feed_dict

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """

        # batch_size = len(train_y)
        # input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        # input_feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch = train_x
        # q_inputs_batch, q_mask_batch = zip(*(train_x[0])) 
        # c_inputs_batch, c_mask_batch = zip(*(train_x[1])) 

        c_inputs_batch = c_inputs_batch.reshape(-1, self.config.max_context_length, 1)
        q_inputs_batch = q_inputs_batch.reshape(-1, self.config.max_question_length, 1)
        # q_mask_batch = np.array(q_mask_batch)
        # c_mask_batch = np.array(c_mask_batch)

        # print('Q', q_inputs_batch.shape)
        # print('QM', q_mask_batch.shape)
        # print('C', c_inputs_batch.shape)
        # print('CM', c_mask_batch.shape)
        # print('S', train_y.shape)

        
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
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        # q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch = valid_x
        
        # c_inputs_batch = c_inputs_batch.reshape(batch_size, self.config.max_context_length, 1)
        # q_inputs_batch = q_inputs_batch.reshape(batch_size, self.config.max_question_length, 1)
        

        # input_feed = self.create_feed_dict(q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch, train_y, 1.0)

        # output_feed = [self.pred_start, self.pred_end]

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
        # input_feed['valid_x'] = valid_x
        q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch = test_x

        # batch_size = q_inputs_batch.shape[0]
        
        c_inputs_batch = c_inputs_batch.reshape(-1, self.config.max_context_length, 1)
        q_inputs_batch = q_inputs_batch.reshape(-1, self.config.max_question_length, 1)
        

        input_feed = self.create_feed_dict(q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch)

        output_feed = [self.pred_start, self.pred_end]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)
        mask = test_x[1]

        a_s = np.minimum(np.argmax(yp, axis=1), mask - 1)
        a_e = np.minimum(np.argmax(yp2, axis=1), mask - 1)

        answers = np.array(zip(a_s, a_e))
        answers.sort(axis=1)

        return answers

        # return (a_s, a_e)

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

    def span_to_sentence(self, span, context):
        words_indices = [context[i] for i in range(span[0], span[1] + 1)]
        return ' '.join([self.rev_vocab[w] for w in words_indices])

    def evaluate_answer(self, session, sample=100, log=True):
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
        test_indices = range(self.test_data['s'].shape[0])
        test_sample_indices = random_sample(test_indices, sample)

        q_batch = self.get_batch_from_indices(self.test_data['q'][0], test_sample_indices)[:,:self.config.max_question_length]
        q_mask_batch = np.clip(self.get_batch_from_indices(self.test_data['q'][1], test_sample_indices), 0, self.config.max_question_length - 1)

        c_batch = self.get_batch_from_indices(self.test_data['c'][0], test_sample_indices)[:,:self.config.max_context_length]
        c_mask_batch = np.clip(self.get_batch_from_indices(self.test_data['c'][1], test_sample_indices), 0, self.config.max_context_length - 1)

        s_batch = self.get_batch_from_indices(self.test_data['s'], test_sample_indices)

        predicted_answers = self.answer(session, (q_batch, q_mask_batch, c_batch, c_mask_batch))
        # print(s_batch)



        for i in range(sample):
            # print("*" * 20)
            # print(test_sample_indices[i])
            context = self.test_data['c'][0][test_sample_indices[i]]
            pred_span = predicted_answers[i]
            actual_span = s_batch[i]
            
            groundtruth = self.span_to_sentence(actual_span, context)
            prediction = self.span_to_sentence(pred_span, context)

            # print(prediction)
            # print(groundtruth)

            f1 += 1.0 * f1_score(prediction, groundtruth) / sample
            em += 1.0 * exact_match_score(prediction, groundtruth) / sample

        # context =[4626, 4, 1207, 27, 2787, 9, 1994, 3, 31852, 239, 6505, 2630, 4, 224, 9, 172, 11018, 4920, 16978, 6, 162, 4, 3, 3127, 3090, 124, 316, 21, 10, 183, 5, 1764, 127, 3370, 3851, 3, 31852, 26748, 2630, 8, 3, 51407, 129, 964, 3847, 7, 52463, 9, 5458, 6, 6448, 4, 3, 11050, 5, 33, 244, 760, 114, 5, 16978, 8, 3, 3783, 18437, 3515, 541, 4552, 6, 4293, 1425, 10518, 31, 3, 5403, 27, 790, 20, 3, 1540, 6, 7135, 16978, 6222, 71, 40, 19917, 27, 3057, 6, 242, 284, 493, 4, 68, 16978, 31, 3, 5403, 27, 157, 9, 34, 5004, 4, 107, 3, 3402, 327, 19, 262, 68, 1868, 16978, 21, 767, 5634, 71, 3, 5468, 6, 242, 284, 1498, 4, 323, 4, 50, 18437, 13, 220, 5634, 6, 14, 5634, 18437, 13, 124, 157, 2234, 104, 3, 18354, 5, 33, 9823, 6, 51161, 6446, 6446, 4, 10, 763, 5, 313, 31, 3, 3465, 658, 4, 13, 982, 24, 10007, 4, 99, 383, 7838, 20, 10, 2589, 8, 99, 9823, 6, 26748, 35157, 7, 47, 1660, 988, 99, 793, 8, 10, 252, 6581, 53261, 7, 3389, 99, 619, 3, 3465, 948, 6]
        # span = (7, 11)
        # print(start_preds, end_preds)
        # print(self.span_to_sentence(span, context))

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def get_batch_from_indices(self, data, indices):
        # print(data)
        # print(indices)
        return data[indices]
        # return [data[i] for i in indices]

    def get_minibatches(self, indices, minibatch_size, shuffle=True):
        data_size = len(indices)
        if shuffle:
            np.random.shuffle(indices)
        batch_indices = []
        for minibatch_start in np.arange(0, data_size, minibatch_size):
            batch_indices.append(indices[minibatch_start:minibatch_start + minibatch_size]) 
        return batch_indices

    def run_epoch(self, sess):
        train_indices = range(self.train_data['s'].shape[0])
        # print("HERE!!!!!!!!!!!", len(train_indices))
        prog = Progbar(target=1 + int(len(train_indices) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch_indices in enumerate(self.get_minibatches(train_indices, self.config.batch_size)):
            q_batch = self.get_batch_from_indices(self.train_data['q'][0], batch_indices)
            q_mask_batch = self.get_batch_from_indices(self.train_data['q'][1], batch_indices)

            c_batch = self.get_batch_from_indices(self.train_data['c'][0], batch_indices)
            c_mask_batch = self.get_batch_from_indices(self.train_data['c'][1], batch_indices)

            s_batch = self.get_batch_from_indices(self.train_data['s'], batch_indices)


            loss = self.optimize(sess, (q_batch, q_mask_batch, c_batch, c_mask_batch), s_batch)
            # losses.append(loss)
            # grad_norms.append(grad_norm)
            # loss = 0
            # print(self.get_batch_from_indices(train['s'], batch_indices))
            prog.update(i + 1, [("train loss", loss)])

        logging.info("Evaluating on development data")
        f1, em = self.evaluate_answer(sess, self.config.evaluate, log=True)
        # print(len(self.vocab))
        # token_cm, entity_scores = self.evaluate_answer(sess, val, dev_set_raw)
        # logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        # logger.debug("Token-level scores:\n" + token_cm.summary())
        # logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        # f1 = entity_scores[-1]
        # return f1

        return loss, f1, em #losses, grad_norms

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
        self.saver = tf.train.Saver()
        best_score = 0. 

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        losses, f1s, ems = [], [], []
        for epoch in range(self.config.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            loss, f1, em = self.run_epoch(session)

            if f1 > best_score:
                best_score = f1
                logging.info("New best score! Saving model in %s", train_dir + "/model.ckpt")
                self.saver.save(session, train_dir + "/model.ckpt")

            losses.append(loss)
            f1s.append(f1)
            ems.append(ems)