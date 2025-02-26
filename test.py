# time series anomaly detection methods
from deepod.models.time_series import TimesNet, COUTA, DeepSVDDTS, DeepIsolationForestTS, TranAD
import pandas as pd
from testbed.utils import data_standardize
import numpy as np

dataset_name = 'MSL'
data_folder = f'dataset/DCdetector_dataset/{dataset_name}/'

labels = np.load(data_folder + f'{dataset_name}_test_label.npy')
X_train_df = pd.DataFrame(np.load(data_folder + f'{dataset_name}_train.npy'))
X_test_df = pd.DataFrame(np.load(data_folder + f'{dataset_name}_test.npy'))
X_train, X_test = data_standardize(X_train_df, X_test_df)

subset_size = 1000
X_train = X_train[:subset_size]
X_test = X_test[:subset_size]
labels = labels[:subset_size]

print(X_train.shape, X_test.shape, labels.shape)

print('Training model...') 
clf = TimesNet(seq_len=10, epoch_steps=-1, stride=1, epochs=2)
clf.fit(X_train)
print('Predicting...')
scores = clf.decision_function(X_test)
print(f'scores shape {scores.shape}')

# evaluation of time series anomaly detection
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment # execute point adjustment for time series ad
print('Evaluating...')
eval_metrics = ts_metrics(labels, scores)
adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
print('Results:')
print('Original:', eval_metrics)
print('Point Adjustment:', adj_eval_metrics)
