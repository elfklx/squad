from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 1.0, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

tf.app.flags.DEFINE_integer("evaluate", 5000, "Number of evaluation samples.")
tf.app.flags.DEFINE_boolean("clip_gradients", True, "Clip gradients?.")

tf.app.flags.DEFINE_integer("max_question_length", 40, "Maximum length of a sentence.")
tf.app.flags.DEFINE_integer("max_context_length", 750, "Maximum context paragraph of a sentence.")

tf.app.flags.DEFINE_integer("label_size", 2, "Size of labels.")
tf.app.flags.DEFINE_integer("hidden_size", 300, "Size of labels.")

tf.app.flags.DEFINE_integer("data_limit", -1, "Limit amount of data.")

tf.app.flags.DEFINE_boolean("load_train_answers", False, "Clip gradients?.")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def get_lines_from_file(filename, sentence=False):
    all_arrays = []   
    with open(filename) as f:
        count = 0
        for line in f:
            if sentence:
                all_arrays.append(line.strip())
            else:
                all_arrays.append(np.fromstring(line, dtype=int, sep=' '))
            if FLAGS.data_limit > -1 and count >= FLAGS.data_limit:
                break
            count += 1
    return all_arrays

def lines_to_padded_np_array(data, max_length):
    padded_seqs = []
    masks = []
    
    for sentence in data:
        sentence_len = len(sentence)
        sentence = map(int, sentence)
        if sentence_len >= max_length:
            padded_seqs.append(np.array(sentence[:max_length]))
            masks.append(max_length)
        else:
            p_len = max_length - sentence_len
            new_sentence = sentence + [0] * p_len
            padded_seqs.append(np.array(new_sentence))
            masks.append(sentence_len)
    return {'data': np.array(padded_seqs), 'mask': np.array(masks)}

class MyData(object):

    def __init__(self):
        train_q_ids = get_lines_from_file(pjoin(FLAGS.data_dir, "new_train.ids.question"))
        train_c_ids = get_lines_from_file(pjoin(FLAGS.data_dir, "new_train.ids.context"))
        train_span = get_lines_from_file(pjoin(FLAGS.data_dir, "new_train.span"))

        # print(train_span)
        
        self.train_q_ids = lines_to_padded_np_array(train_q_ids, FLAGS.max_question_length)
        self.train_c_ids = lines_to_padded_np_array(train_c_ids, FLAGS.max_context_length)
        self.train_span = np.clip(np.array(train_span), 0, FLAGS.max_context_length - 1) 
        self.train_answers = None
        if FLAGS.load_train_answers:
            self.train_answers = get_lines_from_file(pjoin(FLAGS.data_dir, "new_train.answer"), sentence=True)


        dev_q_ids = get_lines_from_file(pjoin(FLAGS.data_dir, "dev.ids.question"))
        dev_c_ids = get_lines_from_file(pjoin(FLAGS.data_dir, "dev.ids.context"))
        dev_span = get_lines_from_file(pjoin(FLAGS.data_dir, "dev.span"))
        
        self.dev_q_ids = lines_to_padded_np_array(dev_q_ids, FLAGS.max_question_length)
        self.dev_c_ids = lines_to_padded_np_array(dev_c_ids, FLAGS.max_context_length)
        self.dev_span = np.clip(np.array(dev_span), 0, FLAGS.max_context_length - 1) 
        self.dev_answers = get_lines_from_file(pjoin(FLAGS.data_dir, "dev.answer"), sentence=True)



def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = MyData()

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    with np.load(embed_path) as data:
        glove_embeddings = np.asfarray(data["glove"], dtype=np.float32)

        encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, config=FLAGS)
        decoder = Decoder(output_size=FLAGS.output_size, config=FLAGS)

        qa = QASystem(encoder, decoder, embeddings=glove_embeddings, config=FLAGS, vocab=(vocab, rev_vocab))

        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        print(vars(FLAGS))
        with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        with tf.Session() as sess:
            load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
            initialize_model(sess, qa, load_train_dir)

            save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
            qa.train(sess, dataset, save_train_dir)

            qa.evaluate_answer(sess, sample=FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
