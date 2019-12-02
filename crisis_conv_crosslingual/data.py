
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

TEXT_COL = 1
TEXT_COL_NAME = 'text'

class DataLoader:
    def __init__(self, args):
        self.args = args


        self.train_labeled_path = args.data_labeled
        self.train_unlabeled_path = args.data_unlabeled
        self.test_path = args.data_test

        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None
        self.y_unlabeled = None

        self.X_test = None

        self.y_cols = []
        self.x_cols = None

        self.age_scaler= None
        self.feature_labelers = {}

    def load(self):

        ds = pd.read_csv(
            self.train_labeled_path,
            parse_dates=['timestamp'],
            dtype={'id':object, 'conversation_id':object,'in_reply_to_status_id':object})
        ds_unlabeled = None
        ds_test = None

        # augment unlabeled instances
        if self.args.num_unlabeled > 0:
            ds_unlabeled = pd.read_csv(self.train_unlabeled_path, dtype=object)

        if self.args.predict:
            with open(self.test_path, encoding="latin-1") as datafile:
                ds_test = pd.read_csv(datafile, dtype=object)
            ds_test['text'].fillna('', inplace=True)

        # fill nan in text
        ds['text'].fillna('', inplace=True)

        # fill y cols
        self.y_cols = self.args.labels

        # for col in self.y_cols:
        #     ds[col].fillna(0, inplace=True)
        #     ds[col] = ds[col].astype(int)
        self.encode_labels(ds)

        # augment features

        if self.args.use_allfeats:
            self.x_cols = ['id', 'text']
        else:
            self.x_cols = ['id', 'text']

        # standarize
        #self.standarize_feats(ds,ds_unlabeled, ds_test)
        self.X_labeled = ds[self.x_cols]
        self.y_labeled = ds[self.y_cols]



        # load unlabeled train set
        if self.args.num_unlabeled > 0:
            #self.standarize_feats(ds_unlabeled)
            self.X_unlabeled = ds_unlabeled[self.x_cols]
            y_train_unlabeled = np.full((ds_unlabeled.shape[0], self.y_labeled.shape[1]), -1)
            self.y_unlabeled = pd.DataFrame(y_train_unlabeled, columns=self.y_cols)

        # load tests set
        if self.args.predict:
            #self.standarize_feats(ds_test)
            self.encode_labels(ds_test)
            self.X_test = ds_test[self.x_cols]
            self.y_test = ds_test[self.y_cols]


    def encode_labels(self, ds, ds_test=None):

        X = ds
        X_all = ds

        if ds_test is not None:
            X_all = X_all.append(ds_test, ignore_index=True, sort=False)


        feats = self.y_cols

        for f in feats:
            X[f].fillna('unk', inplace=True)
            if f not in self.feature_labelers:
                X_all[f].fillna('unk', inplace=True)
                labeler = LabelEncoder()
                labeler.fit(X_all[f].values)
                self.feature_labelers[f] = labeler
            labeler = self.feature_labelers[f]
            X[f] = labeler.transform(X[f].values)

    def standarize_feats(self, ds, ds_unlabeled=None, ds_test=None):
        if not self.args.use_allfeats:
            return

        self.transform_age(ds)
        self.transform_age(ds_unlabeled)
        self.transform_age(ds_test)

        X = ds
        X_all = ds

        if ds_unlabeled is not None:
            X_all = X_all.append(ds_unlabeled, ignore_index=True, sort=False)

        if ds_test is not None:
            X_all = X_all.append(ds_test, ignore_index=True, sort=False)

        if self.age_scaler is None:
            self.age_scaler = StandardScaler()
            self.age_scaler.fit(X_all.age.values.reshape(-1, 1))

        X['age'] = self.age_scaler.transform(X.age.values.reshape(-1, 1))

        feats = [ 'gender', 'married', 'parenthood', 'country', 'reflection', 'duration']

        for f in feats:
            X[f].fillna('unk', inplace=True)
            if f not in self.feature_labelers:
                X_all[f].fillna('unk', inplace=True)
                labeler = LabelEncoder()
                labeler.fit(X_all[f].values)
                self.feature_labelers[f] = labeler
            labeler = self.feature_labelers[f]
            X[f] = labeler.transform(X[f].values)

    def transform_age(self, X):

        if X is None:
            return

        if X.age.dtype == float:
            return

        X.loc[X.age == 'prefer not to say', 'age'] = 0
        X.loc[X.age == '{}', 'age'] = 0
        X.age.fillna(0, inplace=True)
        X['age'] = X.age.apply(lambda x: str(x).replace('years', '').replace(',',''))
        X['age'] = X.age.astype(float)



if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--data-labeled', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--data-unlabeled', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--data-test', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--use-allfeats', action='store_true')
    args = parser.parse_args()
    loader = DataLoader(args)
    loader.load()

    print(f'{loader.X_labeled.shape}')
    print(f'{loader.y_labeled.shape}')
    print(f'{loader.y_labeled.describe()}')