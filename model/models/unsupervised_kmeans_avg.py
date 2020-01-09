import numpy as np
from keras_preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models.base_unsupervised import UnsupervisedBaseModel


class UnsupervisedKmeansAvgBaseModel(UnsupervisedBaseModel):
    def __init__(self, task):
        super(UnsupervisedKmeansAvgBaseModel, self).__init__(task)
        self.num_clusters = 4 # combinations of social and agency
        self.clf_model = KMeans(
            init='k-means++',
            n_clusters=self.num_clusters,
            n_init=10,
            random_state=self.args.random_state
        )

    def augment_features(self, X_text, X_all_feats):

        if not self.args.use_allfeats:
            return X_text

        X_all = np.concatenate(
            [X_text, X_all_feats[:, 2:]],
            axis=1)

        return X_all

    def train(self, X, y=None):
        X, y = self.augment_instances(X, y)

        #X_text = self.text_repr_model.fit_transform(X[:, self.args.TEXT_COL])

        X_text = X[:, self.args.TEXT_COL]

        self.max_features = 4000
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(X_text)
        X_text = self.tokenizer.texts_to_sequences(X_text)
        X_text = self.tokenizer.sequences_to_texts(X_text)

        self.text_rep_model = self.build_fit_w2v(X_text)

        X_text = self.transform_text_to_w2v(self.text_rep_model, X_text)

        X_all_feats = self.augment_features(X_text, X)

        pca = PCA(
            n_components=self.num_clusters,
            random_state=self.args.random_state
        )
        pca.fit(X_all_feats)

        model = KMeans(
            init=pca.components_,
            n_clusters=self.num_clusters,
            n_init=1,
            random_state=self.args.random_state
        )
        model.fit(X_all_feats)

        self.clf_model = model

    def predict(self, X):

        X_text = X[:, self.args.TEXT_COL]

        #X_text = self.text_rep_model.transform(X[:, self.args.TEXT_COL])
        X_text = self.transform_text_to_w2v(self.text_rep_model, X_text)

        X_all_feats = self.augment_features(X_text, X)
        y_pred = self.clf_model.predict(X_all_feats)

        y = y_pred.astype(np.uint8)
        y = np.unpackbits(y)
        y = y.reshape(y_pred.shape[0],8)
        y = y[:, -2:]
        y = y[:, ::-1]

        return y
