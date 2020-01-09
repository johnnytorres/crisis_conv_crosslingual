import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from models.base_unsupervised import UnsupervisedBaseModel


class UnsupervisedKmeansBaseModel(UnsupervisedBaseModel):
    def __init__(self, task):
        super(UnsupervisedKmeansBaseModel, self).__init__(task)
        self.num_clusters = 4 # combinations of social and agency

    def get_text_representation_model(self):
        pass

    def train(self, X, y=None):
        X, y = self.augment_instances(X, y)

        X_text = self.text_repr_model.fit_transform(X[:, self.args.TEXT_COL])

        X_all_feats = self.augment_features(X_text, X)


        # pca = PCA(
        #     n_components=self.num_clusters,
        #     random_state=self.args.random_state
        # )
        # pca.fit(X_all_feats)
        #
        # model = KMeans(
        #     init=pca.components_,
        #     n_clusters=self.num_clusters,
        #     n_init=1,
        #     random_state=self.args.random_state
        # )
        # model.fit(X_all_feats)
        #
        # self.clf_model = model

    def predict(self, X):
        X_text = self.text_repr_model.transform(X[:, self.args.TEXT_COL])
        X_all_feats = self.augment_features(X_text, X)
        y_pred = self.clf_model.predict(X_all_feats)

        y = y_pred.astype(np.uint8)
        y = np.unpackbits(y)
        y = y.reshape(y_pred.shape[0],8)
        y = y[:, -2:]
        y = y[:, ::-1]

        return y
