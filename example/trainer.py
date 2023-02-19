from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn

import lstm_encoder_decoder
import random
"""
load dataset with for pytorch dataloader iterator format
"""
class loadDataset(Dataset):
    def __init__(self, dataframe, target, features):
        self.features = features
        self.target = target
        #print(dataframe[target].values)
        self.y = torch.tensor((dataframe[target].values)).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        return x, self.y[i]


"""
sliding windowed dataset will out put the result for label
"""
def windowed_dataset_1res(y,x, input_window = 5, output_window = 1, stride = 1, num_features = 1,num_out =1):

    '''
    create a windowed dataset

    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model
    : param output_window:    number of future y samples to predict
    : param stide:            spacing between windows
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''

    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1


    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_out])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]
    for ff in np.arange(num_out):
        for ii in np.arange(num_samples):
            start_y = stride * ii + input_window
            end_y = start_y + output_window
            Y[:, ii, ff] = x[start_y:end_y, ff]

    return X, Y

"""
sliding windowed dataset will out put the result for whole 14 features
"""
def windowed_dataset_1(y,x, input_window = 5, output_window = 1, stride = 1, num_features = 1,num_out =1):

    '''
    create a windowed dataset

    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model
    : param output_window:    number of future y samples to predict
    : param stide:            spacing between windows
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''

    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1


    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_out])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]
    for ff in np.arange(num_out):
        for ii in np.arange(num_samples):
            start_y = stride * ii + input_window
            end_y = start_y + output_window
            Y[:, ii, ff] = x[start_y:end_y, ff]

    return X, Y

"""
READ INPUT CSV file to pd type and save to a list
"""
train_file = []
test_file =[]
for i in range(799):
    train_file.append(pd.read_csv('5.0cluster_2_2/input'+str(i)+'.csv'))

#df_train = pd.read_csv("datasets/input2.csv")
#df_test = pd.read_csv("datasets/test.csv")
#print(df_train)
#print(df_train.shape)
train_loader =[]
j=0
for i in train_file:
    batch_size = len(i)
    print(j)
    train_dataset = loadDataset(
        i,
        target=['label'],
        features=['npc_speed'
             ,'ego_speed','location_MHD','angel_MHD', 'clock',
                  'label'],

    )
    train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False))
    j+=1




"""
save sliding windowed dataset from dataloader
"""


print("total number of the training case is ")
print(len(train_loader))


train_loader_windowed_data = []
a=0
for i in train_loader:
   print(a)
   for  x,y in i:

       train_loader_windowed_data.append(windowed_dataset_1(x,y,input_window = 5, output_window = 8 , stride = 1, num_features = 5))
       pass
   a+=1
print("total number of the windowed dataset is ")
print(len(train_loader_windowed_data))


random.shuffle(train_loader_windowed_data)
Xtrain  = train_loader_windowed_data[-1][0]
print(Xtrain)
Ytrain  = train_loader_windowed_data[-1][1]
print(Ytrain)

X_train, Y_train, X_test, Y_test = numpy_to_torch(Xtrain, Ytrain, Xtrain, Ytrain)


print("the shape of one case of Xs is, (first num means num of sliding window, "
      "second means the length of input window, third means the num of feature)")
print(X_train.shape)
#print(X_train)
#print(windowdataset[0][0].shape)
print("the shape of y is (first num means num of sliding window, "
      "second means the length of out window, third means the num of feature)")
print(Y_train.shape)
#print(Y_train)


"""
train model
"""


#define model first with for lstm
model = lstm_encoder_decoder.lstm_seq2seq(input_size = 5, hidden_size = 200)


loss = model.train_model_with_fc(train_loader_windowed_data, n_epochs = 15, target_len = 8, batch_size = 1, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.001)
# Specify a path
PATH = "5.0_cluster2_modelV3.pt"

# Save
torch.save(model.state_dict(), PATH)
