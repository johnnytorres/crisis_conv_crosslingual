from pathlib import Path
from typing import List, Union

import numpy as np
from flair.data import TaggedCorpus, Sentence
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings, CharacterEmbeddings, FlairEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

from models.supervised_fasttext import FastTextModel


glove_embeddings=WordEmbeddings('glove')
fasttext_embeddings=WordEmbeddings('crawl')
twitter_embeddings=WordEmbeddings('twitter')
fflair= FlairEmbeddings('multi-forward')
bflair=FlairEmbeddings('multi-backward')

class BiLstmModel(FastTextModel):
    def __init__(self, task):
        super(BiLstmModel, self).__init__(task)
        # set parameters:
        #self.max_features = 20000
        # cut texts after this number of words (among top max_features most common words)
        #self.maxlen = 80
        #self.batch_size = 32
        self.epochs = 5
        self.clf = None

    # def build_model(self):
    #     print('Build model...')
    #     model = Sequential()
    #     model.add(Embedding(self.max_features, 128, input_length=self.max_len))
    #     model.add(Bidirectional(LSTM(64)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(2, activation='sigmoid'))
    #
    #     # try using different optimizers and different optimizer configs
    #     model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    #
    #     return model

    def save(self, model_file: Union[str, Path]):
        pass

    def save_checkpoint(self, model_file: Union[str, Path], optimizer_state: dict, scheduler_state: dict, epoch: int,
                        loss: float):
        pass

    def train(self, X, y):

        X_text = X[:,self.args.TEXT_COL]
        y = y.flatten()
        #corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03)

        train : List[Sentence] = []

        for tweet, label in zip(X_text, y):
            if tweet == '':
                tweet = 'dummy'
            s : Sentence = Sentence(tweet)
            s.add_label(str(label))
            train.append(s)

        corpus: TaggedCorpus = TaggedCorpus(train, train,train)

        # 2. create the label dictionary
        label_dict = corpus.make_label_dictionary()

        # 3. make a list of word embeddings
        word_embeddings =[
            glove_embeddings,
            #twitter_embeddings,
            # comment in this line to use character embeddings
            #CharacterEmbeddings(),
            # comment in flair embeddings for state-of-the-art results
            # FlairEmbeddings('news-forward'),
            fflair,
            # FlairEmbeddings('news-backward'),
            bflair
        ]

        # 4. initialize document embedding by passing list of word embeddings
        document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                                             hidden_size=512,
                                                                             reproject_words=True,
                                                                             reproject_words_dimension=256,
                                                                             )
        # 5. create the text classifier
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

        # 6. initialize the text classifier trainer
        trainer = ModelTrainer(classifier, corpus)

        self.model = trainer.model
        self.model.save = self.save
        self.model.save_checkpoint = self.save_checkpoint

        # 7. start the training
        trainer.train('../data/ecuador_earthquake_2016/models',
                      learning_rate=0.1,
                      mini_batch_size=32,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=5)

        self.clf = classifier

    def predict(self, X):

        X_text = X[:, self.args.TEXT_COL]

        sentences = []

        for tweet in X_text:

            if tweet == '':
                tweet = 'dummy'

            s: Sentence = Sentence(tweet)
            #s.add_label(label)
            sentences.append(s)

        sentences= self.clf.predict(sentences)

        predictions = []

        for s in sentences:
            p = s.labels[0].value
            predictions.append(int(p))

        return predictions


if __name__ == '__main__':


    np.random.seed(1)

    #ds =

    X_text = X[:, self.args.TEXT_COL]
    y = y.flatten()
    # corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.CONLL_03)

    train: List[Sentence] = []

    for tweet, label in zip(X_text, y):
        if tweet == '':
            tweet = 'dummy'
        s: Sentence = Sentence(tweet)
        s.add_label(label)
        train.append(s)

    corpus: TaggedCorpus = TaggedCorpus(train)

    # 2. create the label dictionary
    label_dict = corpus.make_label_dictionary()

    # 3. make a list of word embeddings
    word_embeddings = [WordEmbeddings('glove'),

                       # comment in flair embeddings for state-of-the-art results
                       # FlairEmbeddings('news-forward'),
                       # FlairEmbeddings('news-backward'),
                       ]

    # 4. initialize document embedding by passing list of word embeddings
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                                         hidden_size=512,
                                                                         reproject_words=True,
                                                                         reproject_words_dimension=256,
                                                                         )
