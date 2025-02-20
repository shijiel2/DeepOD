# time series anomaly detection methods
from deepod.models.time_series import TimesNet, AnomalyTransformer
import pandas as pd
from testbed.utils import data_standardize
import numpy as np

dataset_name = 'SMAP'
data_folder = f'dataset/DCdetector_dataset/{dataset_name}/'

labels = np.load(data_folder + f'{dataset_name}_test_label.npy')
X_train_df = pd.DataFrame(np.load(data_folder + f'{dataset_name}_train.npy'))
X_test_df = pd.DataFrame(np.load(data_folder + f'{dataset_name}_test.npy'))
X_train, X_test = data_standardize(X_train_df, X_test_df)


# train_df = pd.read_csv('dataset/DCdetector_dataset/PSM/PSM_train.csv', sep=',', index_col=0)
# test_df = pd.read_csv('dataset/DCdetector_dataset/PSM/PSM_test.csv', sep=',', index_col=0)
# labels = pd.read_csv('dataset/DCdetector_dataset/PSM/PSM_test_label.csv', sep=',', index_col=0).values
# X_train, X_test = data_standardize(train_df, test_df)

print(X_train.shape, X_test.shape, labels.shape)

print('Training model...') 
clf = TimesNet(seq_len=10, epoch_steps=-1, stride=1)
clf.fit(X_train)
print('Predicting...')
scores = clf.decision_function(X_test)

# evaluation of time series anomaly detection
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment # execute point adjustment for time series ad
print('Evaluating...')
eval_metrics = ts_metrics(labels, scores)
adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
print('Results:')
print('Original:', eval_metrics)
print('Point Adjustment:', adj_eval_metrics)
