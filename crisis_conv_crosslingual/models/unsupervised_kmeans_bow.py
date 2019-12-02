import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from models.base_unsupervised import UnsupervisedBaseModel


class UnsupervisedKmeansBowModel(UnsupervisedBaseModel):
    def __init__(self, task):
        super(UnsupervisedKmeansBowModel, self).__init__(task)
        self.num_clusters = 4 # combinations of social and agency
        self.text_repr_model = self.get_text_representation_model()
        self.clf_model = KMeans(
            init='k-means++',
            n_clusters=self.num_clusters,
            n_init=10,
            random_state=self.args.random_state
        )

    def augment_features(self, X_text, X_all_feats):

        if not self.args.use_allfeats:
            return X_text.toarray()

        age = X_all_feats[:, 2].reshape(-1, 1)
        gender = X_all_feats[:, 3].reshape(-1, 1)
        married = X_all_feats[:, 4].reshape(-1, 1)
        parenthood = X_all_feats[:, 5].reshape(-1, 1)
        country = X_all_feats[:, 6].reshape(-1, 1)
        reflection = X_all_feats[:, 7].reshape(-1, 1)
        duration = X_all_feats[:, 8].reshape(-1, 1)

        X_all = np.concatenate(
            [X_text.toarray(), age, gender, married,parenthood,country,reflection, duration],
            axis=1)

        return X_all

    def get_text_representation_model(self):
        steps = []
        vectorizer = TfidfVectorizer(
            ngram_range=(1, self.args.ngrams),
            min_df=5,
            max_df=0.5,
            stop_words="english",
            use_idf=False
        )
        steps.append(('vec', vectorizer))
        repr_model = Pipeline(steps)
        return repr_model

    def train(self, X, y=None):
        X, y = self.augment_instances(X, y)
        X_text = self.text_repr_model.fit_transform(X[:, self.args.TEXT_COL])
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
        X_text = self.text_repr_model.transform(X[:, self.args.TEXT_COL])
        X_all_feats = self.augment_features(X_text, X)
        y_pred = self.clf_model.predict(X_all_feats)

        y = y_pred.astype(np.uint8)
        y = np.unpackbits(y)
        y = y.reshape(y_pred.shape[0],8)
        y = y[:, -2:]
        y = y[:, ::-1]

        return y
