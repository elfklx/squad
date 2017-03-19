from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from util import Progbar, minibatches

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

tf.app.flags.DEFINE_integer("evaluate", 5000, "Number of evaluation samples.")
tf.app.flags.DEFINE_boolean("clip_gradients", True, "Clip gradients?.")

tf.app.flags.DEFINE_integer("max_question_length", 40, "Maximum length of a sentence.")
tf.app.flags.DEFINE_integer("max_context_length", 750, "Maximum context paragraph of a sentence.")

tf.app.flags.DEFINE_integer("label_size", 2, "Size of labels.")
tf.app.flags.DEFINE_integer("hidden_size", 300, "Size of labels.")

tf.app.flags.DEFINE_integer("data_limit", -1, "Limit amount of data.")

tf.app.flags.DEFINE_boolean("load_train_answers", False, "Clip gradients?.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")

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
    # print(model)
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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data

def get_array_from_string_array(string_array):
    all_arrays = []  
    for line in string_array:
        all_arrays.append(np.fromstring(line, dtype=int, sep=' '))
    return all_arrays

def pad_sequences(data, max_length):
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
    return (np.array(padded_seqs), np.array(masks))

def lines_to_padded_np_array(data, max_length):
    padded_seqs = []
    masks = []
    
    for sentence in data:
        # print(sentence)
        
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


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    # contexts = get_array_from_string_array(dataset[0])
    # questions = get_array_from_string_array(dataset[1])    
    # uuids = dataset[2]

    # questions = pad_sequences(questions, FLAGS.max_question_length)
    # contexts = pad_sequences(contexts, FLAGS.max_context_length)  

    # predicted_answers = model.answer(sess, (questions[0], questions[1], contexts[0], contexts[1]))

    # for i, v in enumerate(predicted_answers):
    #     # print(contexts[0][i])
    #     s = model.span_to_sentence(v, contexts[0][i])
    #     answers[uuids[i]] = model.span_to_sentence(v, contexts[0][i])
    #     # print(s)
    # print(dataset[0])

    contexts = get_array_from_string_array(dataset[0])
    questions = get_array_from_string_array(dataset[1])    
    uuids = dataset[2]

    questions = lines_to_padded_np_array(questions, FLAGS.max_question_length)
    contexts = lines_to_padded_np_array(contexts, FLAGS.max_context_length)  



    test_indices = range(questions['data'].shape[0])
    print('Number of questions:', len(test_indices))
    prog = Progbar(target=1 + int(len(test_indices) / FLAGS.batch_size))
    for i, batch_indices in enumerate(model.get_minibatches(test_indices, FLAGS.batch_size, shuffle=False)):
        q_batch = model.get_batch_from_indices(questions['data'], batch_indices)
        q_mask_batch = model.get_batch_from_indices(questions['mask'], batch_indices)

        c_batch = model.get_batch_from_indices(contexts['data'], batch_indices)
        c_mask_batch = model.get_batch_from_indices(contexts['mask'], batch_indices)

        uuids_batch = model.get_batch_from_indices(uuids, batch_indices)

        # Answer batchp
        predicted_answers = model.answer(sess, (q_batch, q_mask_batch, c_batch, c_mask_batch))

        # print('HERE', len(predicted_answers))
        
        for j in range(len(batch_indices)):
            context = c_batch[j]
            pred_span = predicted_answers[j]
            
            answers[uuids_batch[j]] = model.span_to_sentence(pred_span, context)
        
        prog.update(i + 1)

    # print(answers)
    print('Number of answers:', len(answers))

    return answers


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


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    with np.load(embed_path) as data:
        glove_embeddings = np.asfarray(data["glove"], dtype=np.float32)
        
        print(vars(FLAGS))
        with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # ========= Load Dataset =========
        # You can change this code to load dataset in your own way

        dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
        dev_filename = os.path.basename(FLAGS.dev_path)
        context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
        dataset = (context_data, question_data, question_uuid_data)

        # ========= Model-specific =========
        # You must change the following code to adjust to your model

        encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, config=FLAGS)
        decoder = Decoder(output_size=FLAGS.output_size, config=FLAGS)

        qa = QASystem(encoder, decoder, embeddings=glove_embeddings, config=FLAGS, vocab=(vocab, rev_vocab))

        with tf.Session() as sess:
            train_dir = get_normalized_train_dir(FLAGS.train_dir)
            initialize_model(sess, qa, train_dir)
            answers = generate_answers(sess, qa, dataset, rev_vocab)

            # write to json file to root dir
            with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
