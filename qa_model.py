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
        self.config = config

    def encode(self, inputs, masks, encoder_state_input, embeddings):
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
        question_input = tf.nn.embedding_lookup(embeddings, questions)
        context_input = tf.nn.embedding_lookup(embeddings, contexts)
        
        q_mask, c_mask = masks

        # Encode question with GRU
        with tf.variable_scope('q_decoder') as scope:
            q_cell = tf.nn.rnn_cell.GRUCell(self.config.state_size)
            q_outputs, q_output_state = tf.nn.dynamic_rnn(q_cell,
                                                          question_input,
                                                          sequence_length=q_mask, 
                                                          dtype=tf.float32)

        # Encode context paragraph with GRU
        with tf.variable_scope('p_decoder') as scope:
            p_cell = tf.nn.rnn_cell.GRUCell(self.config.state_size)
            c_outputs, c_output_state = tf.nn.dynamic_rnn(p_cell,
                                                          context_input,
                                                          sequence_length=c_mask, 
                                                          dtype=tf.float32)

        # Make correct shape for multiplication       
        reshaped_q = tf.expand_dims(q_output_state, -1)

        # X q
        combined_encoding = tf.reshape(tf.batch_matmul(c_outputs, reshaped_q), [-1, self.config.max_context_length])

        print("Done adding encode definition: ", combined_encoding)        
        return combined_encoding


class Decoder(object):
    def __init__(self, output_size, config):
        self.output_size = output_size
        self.config = config

    def decode(self, knowledge_rep, dropout_rate):
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
        return {"start": self.decode_start(knowledge_rep, dropout_rate), 
                "end": self.decode_end(knowledge_rep, dropout_rate)}

    def decode_start(self, knowledge_rep, dropout_rate):
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('decoder_start') as scope:
            # W = tf.get_variable(name="W", shape=(self.config.max_context_length, self.config.hidden_size), dtype=tf.float32, initializer=xavier_initializer) #
            # b1 = tf.Variable(tf.zeros((self.config.hidden_size,)), name="b1")
            # U = tf.get_variable(name="U", shape=(self.config.hidden_size, self.config.max_context_length), dtype=tf.float32, initializer=xavier_initializer)
            # b2 = tf.Variable(tf.zeros((self.config.max_context_length)), name="b2")
        
            # h = tf.tanh(tf.matmul(knowledge_rep, W) + b1)
            # h_drop = tf.nn.dropout(h, dropout_rate)
            # pred = tf.matmul(h_drop, U) + b2

            # print("Done adding decode start definition:", pred)
            # return pred
            return knowledge_rep  

    def decode_end(self, knowledge_rep, dropout_rate):
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('decoder_end') as scope:
            # W = tf.get_variable(name="W", shape=(self.config.max_context_length, self.config.hidden_size), dtype=tf.float32, initializer=xavier_initializer) #
            # b1 = tf.Variable(tf.zeros((self.config.hidden_size,)), name="b1")
            # U = tf.get_variable(name="U", shape=(self.config.hidden_size, self.config.max_context_length), dtype=tf.float32, initializer=xavier_initializer)
            # b2 = tf.Variable(tf.zeros((self.config.max_context_length)), name="b2")
        
            # h = tf.tanh(tf.matmul(knowledge_rep, W) + b1)
            # h_drop = tf.nn.dropout(h, dropout_rate)
            # pred = tf.matmul(h_drop, U) + b2

            # print("Done adding decode end definition:", pred)
            # return pred
            return knowledge_rep

class QASystem(object):
    def __init__(self, encoder, decoder, embeddings, config, vocab, *args):
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
           
        # ==== set up training/updating procedure ====
            self.add_training_op()
            self.saver = tf.train.Saver()

        logging.info("Done setup!!!!!!!!!!!")


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoding = self.encoder.encode(inputs=(self.question_input_placeholder, self.context_input_placeholder), 
                                       masks=(self.question_masks_placeholder, self.context_masks_placeholder),
                                       encoder_state_input=None,
                                       embeddings=self.embeddings)

        self.pred = self.decoder.decode(encoding, self.dropout_placeholder)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            epsilon = tf.constant(value=1e-50)

            modified_pred_start = self.pred['start'] + tf.log(epsilon + tf.sequence_mask(lengths=self.context_masks_placeholder, maxlen=self.config.max_context_length, dtype=tf.float32))
            modified_pred_end = self.pred['end'] + tf.log(epsilon + tf.sequence_mask(lengths=self.context_masks_placeholder, maxlen=self.config.max_context_length, dtype=tf.float32))

            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_start, self.labels_placeholder[:,0])
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(modified_pred_end, self.labels_placeholder[:,1])

            self.loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            self.embeddings = tf.Variable(self.pretrained_embeddings, name="L")

        logging.info("Done adding embeddings")

    def add_placeholders(self): 
        self.question_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_question_length), name="questions")
        self.question_masks_placeholder = tf.placeholder(tf.int32, shape=(None,), name="question_masks")        

        self.context_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_context_length), name="contexts")
        self.context_masks_placeholder = tf.placeholder(tf.int32, shape=(None,), name="context_masks")

        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.label_size), name="labels")

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")

        logging.info("Done adding placeholders")

    def add_training_op(self):
        self.optimizer = (get_optimizer(self.config.optimizer))(learning_rate=self.config.learning_rate)

        grad_list = self.optimizer.compute_gradients(self.loss)
        grads = [g for g, v in grad_list]
        if self.config.clip_gradients:            
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
            grad_list = [(grads[i], grad_list[i][1]) for i in xrange(len(grads))]
        self.grad_norm = tf.global_norm(grads)        
        self.train_op = self.optimizer.apply_gradients(grad_list)

    def create_feed_dict(self, q_inputs_batch=None, q_mask_batch=None, c_inputs_batch=None, c_mask_batch=None, labels_batch=None, dropout=1):
        feed_dict = {}
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
        q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch = train_x
        input_feed = self.create_feed_dict(q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch, train_y, self.config.dropout)
        output_feed = [self.train_op, self.loss, self.grad_norm]

        _, loss, g_norm = session.run(output_feed, input_feed)
        print('GRADIENT NORM: ', g_norm)
        return loss

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

        # print('HERE', test_x)

        q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch = test_x
        input_feed = self.create_feed_dict(q_inputs_batch, q_mask_batch, c_inputs_batch, c_mask_batch)

        output_feed = [self.pred['start'], self.pred['end']]

        start_pred, end_pred = session.run(output_feed, input_feed)

        start_pred = start_pred + np.array([ [0.0] * m + [-1e50] * (self.config.max_context_length - m) for m in c_mask_batch])
        end_pred = end_pred + np.array([ [0.0] * m + [-1e50] * (self.config.max_context_length - m) for m in c_mask_batch])

        # np.set_printoptions(threshold='nan')
        # print(start_pred[0])
        # print(end_pred[0])



        return (start_pred, end_pred)

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)



        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        # np.set_printoptions(threshold='nan')
        # print(a_s)
        # print(a_e)

        answers = np.array(zip(a_s, a_e))
        answers.sort(axis=1)

        # print(answers)
        return answers

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
        # np.set_printoptions(threshold='nan')
        # print(context)
        words_indices = [context[i] for i in range(span[0], span[1] + 1)]
        return ' '.join([self.rev_vocab[w] for w in words_indices])

    def evaluate_answer(self, session, sample=100, log=False, dataset=None):
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
        if dataset is not None:
            self.dataset = dataset

        f1 = 0.
        em = 0.

        # useDev = True

        test_indices = range(sample)
                
        prog = Progbar(target=1 + int(len(test_indices) / self.config.answer_batch_size))
        losses, grad_norms = [], []
        for i, batch_indices in enumerate(self.get_minibatches(test_indices, self.config.answer_batch_size, shuffle=False)):
            if self.config.useDev:      
                q_batch = self.get_batch_from_indices(self.dataset.dev_q_ids['data'], batch_indices)
                q_mask_batch = self.get_batch_from_indices(self.dataset.dev_q_ids['mask'], batch_indices)

                c_batch = self.get_batch_from_indices(self.dataset.dev_c_ids['data'], batch_indices)
                c_mask_batch = self.get_batch_from_indices(self.dataset.dev_c_ids['mask'], batch_indices)

                s_batch = self.get_batch_from_indices(self.dataset.dev_span, batch_indices)

                groundtruths = self.get_batch_from_indices(self.dataset.dev_answers, batch_indices)
            else:
                q_batch = self.get_batch_from_indices(self.dataset.train_q_ids['data'], batch_indices)
                q_mask_batch = self.get_batch_from_indices(self.dataset.train_q_ids['mask'], batch_indices)

                c_batch = self.get_batch_from_indices(self.dataset.train_c_ids['data'], batch_indices)
                c_mask_batch = self.get_batch_from_indices(self.dataset.train_c_ids['mask'], batch_indices)

                s_batch = self.get_batch_from_indices(self.dataset.train_span, batch_indices)

                groundtruths = self.get_batch_from_indices(self.dataset.train_answers, batch_indices)

            # Answer batch
            predicted_answers = self.answer(session, (q_batch, q_mask_batch, c_batch, c_mask_batch))

            # print('HERE', len(predicted_answers))
            
            for j in range(len(batch_indices)):
                # print("*" * 20)
                # print(test_sample_indices[i])
                # print(test_sample_indices[i])
                context = c_batch[j]
                pred_span = predicted_answers[j]
                # actual_span = s_batch[j]

                groundtruth = groundtruths[j]
                prediction = self.span_to_sentence(pred_span, context)

                # print('Pred:', prediction)
                # print('True:', groundtruth)
                # print(context)
                # print(self.train_answers[test_sample_indices[i]])

                f1 += 100.0 * f1_score(prediction, groundtruth) / sample
                em += 100.0 * exact_match_score(prediction, groundtruth) / sample
            
            prog.update(i + 1)

        
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def get_batch_from_indices(self, data, minibatch_idx):
        return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

    def get_minibatches(self, indices, minibatch_size, shuffle=True):
        data_size = len(indices)
        if shuffle:
            np.random.shuffle(indices)
        batch_indices = []
        for minibatch_start in np.arange(0, data_size, minibatch_size):
            batch_indices.append(indices[minibatch_start:minibatch_start + minibatch_size]) 
        return batch_indices

    def run_epoch(self, sess):
        train_indices = range(self.dataset.train_q_ids['data'].shape[0])
        
        prog = Progbar(target=1 + int(len(train_indices) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch_indices in enumerate(self.get_minibatches(train_indices, self.config.batch_size)):
            # Batch of questions
            q_batch = self.get_batch_from_indices(self.dataset.train_q_ids['data'], batch_indices)
            q_mask_batch = self.get_batch_from_indices(self.dataset.train_q_ids['mask'], batch_indices)

            # Batch of context paragraphs
            c_batch = self.get_batch_from_indices(self.dataset.train_c_ids['data'], batch_indices)
            c_mask_batch = self.get_batch_from_indices(self.dataset.train_c_ids['mask'], batch_indices)

            # Batch of spans for answer
            s_batch = self.get_batch_from_indices(self.dataset.train_span, batch_indices)

            # One optimization step
            loss = self.optimize(sess, (q_batch, q_mask_batch, c_batch, c_mask_batch), s_batch)
            
            prog.update(i + 1, [("train loss", loss)])

        logging.info("Evaluating on development data")
        f1, em = self.evaluate_answer(sess, sample=self.config.evaluate, log=True)
        
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

        self.dataset = dataset
        best_score = 0.0

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
