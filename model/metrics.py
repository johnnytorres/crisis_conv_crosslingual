
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

ds = pd.read_csv('../results/predictions.csv')
ds.info()

# calc metrics
labels =['crisis_related']
ds = ds[ds.set == 'cv']
pred_labels = [l + '_pred' for l in labels]
y_true = ds[labels]
y_pred = ds[pred_labels]
metrics = precision_recall_fscore_support(y_true, y_pred)
print(metrics)
