from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from models.unsupervised_kmeans_bow import UnsupervisedKmeansBowModel


class UnsupervisedKmeansTfidfModel(UnsupervisedKmeansBowModel):
    def __init__(self, task):
        super(UnsupervisedKmeansTfidfModel, self).__init__(task)

    def get_text_representation_model(self):
        steps = []
        vectorizer = TfidfVectorizer(
            ngram_range=(1, self.args.ngrams),
            min_df=5,
            max_df=0.5,
            stop_words="english",
            use_idf=True
        )
        steps.append(('vec', vectorizer))
        repr_model = Pipeline(steps)
        return repr_model

