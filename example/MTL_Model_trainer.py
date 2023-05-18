import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import lstm_encoder_decoder

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
def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    
    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 

    '''

    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch


def windowed_dataset(y,x, input_window = 5, output_window = 1, stride = 0, num_features = 1,num_out =14):

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
    #print(L)
    num_samples = (L - input_window - output_window) // stride + 1
    #print(num_samples)


    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([input_window, 1, num_out])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]
    #print(X.shape)
    for ff in np.arange(num_out):
        for ii in np.arange(1):
            start_y = stride * ii + 4
            end_y = start_y + input_window
            Y[:, ii, ff] = x[start_y:end_y, ff]


    return X, Y

def windowed_dataset_v2(y,x, input_window = 5, output_window = 1, stride = 1, num_features = 16,num_out =14):

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
    num_samples = (L - input_window - output_window) // stride


    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_out])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]
    for ff in np.arange(num_out):
        for ii in np.arange(num_samples):
            start_y = stride * ii + input_window -1
            end_y = start_y + output_window
            Y[:, ii, ff] = x[start_y:end_y, ff]

    return X, Y




train_file = []
test_file =[]
for i in range(800):
    train_file.append(pd.read_csv('post_dataset/trial'+str(i)+'.csv'))
train_loader =[]

for i in train_file:
    batch_size = len(i)

    train_dataset = loadDataset(
        i,
        target=["predict_NPC_speed","predict_NPC_Angle_X","predict_NPC_Angle_Y","predict_NPC_Angle_Z","predict_NPC_location_X","predict_NPC_location_Y","predict_NPC_location_Z",
        "predict_AV_speed","predict_AV_Angle_X","predict_AV_Angle_Y","predict_AV_Angle_Z","predict_AV_location_X","predict_AV_location_Y","predict_AV_location_Z"],

        features=["NPC_speed", "NPC_Angle_X","NPC_Angle_Y","NPC_Angle_Z","NPC_location_X", "NPC_location_Y", "NPC_location_Z",
         "AV_speed", "AV_angle_X", "AV_angle_Y", "AV_angle_Z","AV_location_X", "AV_location_Y", "AV_location_Z", "NPC_target_speed","NPC_turning_cmd"],


    )
    train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False))

train_loader_windowed_data = []

for i in train_loader:

   for  x,y in i:

       train_loader_windowed_data.append(windowed_dataset_v2(x,y,input_window = 5, output_window = 1 , stride = 1, num_features = 16))
       pass

print("total number of the windowed dataset is ")
print(len(train_loader_windowed_data))
#print("test")
Xtrain  = train_loader_windowed_data[0][0]


Ytrain  = train_loader_windowed_data[0][1]


model = lstm_encoder_decoder.lstm_seq2seq(input_size = 16, hidden_size = 200)


loss = model.train_model_with_fc(train_loader_windowed_data, n_epochs = 20, target_len = 1, batch_size = 1, training_prediction = 'recursive',learning_rate = 0.01)
# Specify a path
PATH = "predict_model_MTLtest1.pt"

# Save
torch.save(model.state_dict(), PATH)
