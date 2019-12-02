from models.supervised_bilstm import BiLstmModel
from models.supervised_cnn import CnnModel
from models.supervised_cnn_lstm import CnnLstmModel
from models.supervised_fasttext import FastTextModel
from models.supervised_lstm import LstmModel
from models.unsupervised_kmeans_avg import UnsupervisedKmeansAvgBaseModel
from models.unsupervised_kmeans_bow import UnsupervisedKmeansBowModel
from models.supervised_logistic import LogisticModel
from models.unsupervised_kmeans_tfidf import UnsupervisedKmeansTfidfModel


def get_model(task):
    if task.args.model == 'lr':
        return LogisticModel(task)
    if task.args.model == 'fasttext':
        return FastTextModel(task)
    if task.args.model == 'kmeansbow':
        return UnsupervisedKmeansBowModel(task)
    if task.args.model == 'kmeanstfidf':
        return UnsupervisedKmeansTfidfModel(task)
    if task.args.model == 'kmeansavg':
        return UnsupervisedKmeansAvgBaseModel(task)
    if task.args.model == 'cnn':
        return CnnModel(task)
    if task.args.model == 'lstm':
        return LstmModel(task)
    if task.args.model == 'bilstm':
        return BiLstmModel(task)
    if task.args.model == 'cnnlstm':
        return CnnLstmModel(task)
    raise NotImplemented('model not implemented')
