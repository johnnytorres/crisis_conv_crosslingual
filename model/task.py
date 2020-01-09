
import os
import uuid
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from data import DataLoader
from models.factory import get_model


class ClassificationTask:
    def __init__(self, args):
        self.args = args
        self.args.run_id = str(uuid.uuid4())
        self.args.initial_timestamp = datetime.now().timestamp()
        self.args.TEXT_COL = 1
        self.dataset = DataLoader(self.args)
        self.output_path = args.output_file

        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        print("PARAMS: {}".format(self.args))

        # set random state
        np.random.seed(args.random_state)

    def split_dataset(self):

        X = self.dataset.X_labeled.values
        y = self.dataset.y_labeled.values

        if self.args.predict:
            X_train, y_train = X, y
            X_test, y_test = self.dataset.X_test.values, self.dataset.y_test.values
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2,random_state=self.args.random_state)

        return X_train, X_test, y_train, y_test

    def run(self):

        self.dataset.load()

        X_train, X_test,  y_train, y_test = self.split_dataset()

        logging.info("Train data: {}".format(X_train.shape))
        logging.info("Test data: {}".format(X_test.shape))

        labels = self.dataset.y_cols
        results = None
        k_folds = self.args.kfolds
        scores = []
        best_model = None
        best_score = 0
        cv = KFold(k_folds, random_state=self.args.random_state)

        for k, fold in enumerate(cv.split(X_train, y_train)):

            logging.info('training fold {}'.format(k))
            train, valid = fold
            X_kfold, X_valid = X_train[train], X_train[valid]
            y_kfold, y_valid = y_train[train], y_train[valid]

            model = get_model(self)
            model.train(X_kfold, y_kfold)
            y_pred = model.predict(X_valid)

            score = precision_recall_fscore_support(y_valid, y_pred, average='macro')
            score = score[2] #F1
            scores.append(score)
            print(f"CV {k} F1: {score}")

            if score > best_score:
                best_score = score
                best_model = model

            y_pred = pd.DataFrame(y_pred, columns=labels)
            y_valid = pd.DataFrame(y_valid, columns=labels)
            results_df = y_valid.merge(y_pred, left_index=True,right_index=True, suffixes=('', '_pred'))
            results_df['kfold'] = k
            results_df['set'] = 'cv'
            results_df['id'] = X_valid[:,0]
            results_df['timestamp'] = datetime.now().timestamp()
            results_df['run_id'] = self.args.run_id
            results = results_df if results is None else results.append(results_df, ignore_index=True)

        # predict on tests set
        y_pred = best_model.predict(X_test)


        score = precision_recall_fscore_support(y_test, y_pred, average='macro')
        score = score[2] # f1
        scores = np.array(scores)
        print(f"CV F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print(f"Test F1: %0.2f" % (score))

        y_pred = pd.DataFrame(y_pred, columns=labels)
        y_test = pd.DataFrame(y_test, columns=labels)

        results_df = y_test.merge(y_pred, left_index=True,right_index=True, suffixes=('', '_pred'))
        results_df['kfold'] = 0
        results_df['set'] = 'test'
        results_df['id'] = X_test[:, 0]
        results_df['timestamp'] = datetime.now().timestamp()
        results_df['run_id'] = self.args.run_id
        results = results.append(results_df, ignore_index=True)
        write_header = not os.path.exists(self.output_path)
        # save results
        with open(self.output_path, 'a') as f:
            results.to_csv(path_or_buf=f, index=False, header= write_header)

        # save hyperparams
        self.args.final_timestamp = datetime.now().timestamp()
        filepath = os.path.splitext(self.output_path)[0] + '.json'
        with open(filepath, 'a') as f:
            config = json.dumps(self.args.__dict__)
            f.write(config +'\r')



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('initializing task...')
    parser = ArgumentParser()
    parser.add_argument('--data-labeled', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--data-unlabeled', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--data-test', type=lambda x: os.path.expanduser(x))
    parser.add_argument(
        '--model',
        choices=[
            'lr',
            'fasttext',
            'kmeansbow',
            'kmeanstfidf',
            'kmeansavg',
            'cnn',
            'lstm',
            'bilstm',
            'cnnlstm',
            'semikmeans'
        ],
        default='lr'
    )
    parser.add_argument('--labels', type=lambda x: x.split(','))
    parser.add_argument('--use-allfeats', action='store_true')
    parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--kfolds', type=int, default=2)
    parser.add_argument('--ngrams', type=int, default=1)
    parser.add_argument('--embeddings-size', type=int, default=50)
    parser.add_argument('--embeddings-path', type=str, default=None)
    parser.add_argument('--random-state', type=int, default=1)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--output-file', type=lambda x:os.path.expanduser(x), default='../results/predictions.csv')
    task = ClassificationTask(parser.parse_args())
    task.run()
    logging.info('task finished...[ok]')








