import torch
import numpy as np
import sys
import gzip
import pickle
import math

from cnn_functions import CNN
from load_data import Load_data_ind

modelFile  = sys.argv[1]
testdata = sys.argv[2]
predFile  = sys.argv[3]

print('\nModel file: ', modelFile, '\n')
print('Test data: ', testdata, '\n')
print('Prediction result: ', predFile, '\n')

with gzip.open(modelFile, 'rb') as f:
    classifier = pickle.load(f)

test_dataset = Load_data_ind(testdata)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

y_true, y_pred = [], []
classifier.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_true.extend(y_batch.tolist())
        output = classifier(x_batch)
        y_pred.extend(output.tolist())

tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('/')[-1].split('.')[0]+'.'+testdata.split('/')[-1].split('.')[1]).readlines()]

with open(predFile, 'w') as fout:
    for i in range(len(y_true)):
        fout.write(tids[i] + '\t' + str(y_pred[i]) + '\n')
