import os
import logging
import argparse
import numpy as np
import tensorflow as tf

from keras_preprocessing.text import Tokenizer
from tqdm import tqdm

from data import DataLoader


class EmbeddingsBuilder:
    def __init__(self, args):
        logging.info('initializing...')
        self.args = args
        self.dataset = DataLoader(self.args)
        self.embeddings_path = args.embeddings_path
        self.small_embeddings_path = os.path.splitext(self.embeddings_path)[0] + '_small.vec'
        logging.info('initializing...[ok]')

    def build_embedding(self, vocab_dict):
        """
        Load embedding vectors from a .txt file.
        Optionally limit the vocabulary to save memory. `vocab` should be a set.
        """
        num_words = len(vocab_dict)
        num_found = 0

        with open(self.small_embeddings_path, 'w') as out_file:
            with tf.gfile.GFile(self.embeddings_path) as f:
                header =next(f)
                num_embeddings, embeddings_dim = header.split(' ')
                num_embeddings = int(num_embeddings)
                out_file.write(header)
                for _, line in tqdm(enumerate(f), 'loading embeddings', total=num_embeddings):
                    tokens = line.rstrip().split(" ")
                    word = tokens[0]
                    if word in vocab_dict:
                        num_found += 1
                        out_file.write(line)

        tf.logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_found, num_words))

    def run(self):
        self.dataset.load()

        X = self.dataset.X_train_labeled['moment'].values
        X = np.append(X, self.dataset.X_train_unlabeled['moment'].values, axis=0)
        X = np.append(X, self.dataset.X_test['moment'].values, axis=0)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)

        self.build_embedding(tokenizer.word_index)





if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('initializing task...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/claff-happydb')
    parser.add_argument('--embeddings-path', type=str, default=None)
    parser.add_argument('--num-unlabeled', type=int, default=1000)
    parser.add_argument('--use-allfeats', action='store_true', default=False)
    parser.add_argument('--predict', action='store_true', default=True)
    builder = EmbeddingsBuilder(args=parser.parse_args())
    builder.run()
    logging.info('task finished...[ok]')








