import copy
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
import lstm_encoder_decoder_origin
import csv
from sklearn.metrics import r2_score
from torchmetrics import R2Score
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


def windowed_dataset_v2(y,x, input_window = 5, output_window = 1, stride = 1, num_features = 16,num_out =8):

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
for i in range(1):
    train_file.append(pd.read_csv('post_dataset/trial'+str(302)+'.csv'))
train_loader =[]

for i in train_file:
    batch_size = len(i)

    train_dataset = loadDataset(
        i,
         target=["predict_NPC_speed","predict_NPC_location_X","predict_NPC_location_Y","predict_NPC_location_Z",
        "predict_AV_speed","predict_AV_location_X","predict_AV_location_Y","predict_AV_location_Z"],

        features=["NPC_speed", "NPC_location_X", "NPC_location_Y", "NPC_location_Z",
         "AV_speed","AV_location_X", "AV_location_Y", "AV_location_Z", "NPC_target_speed","NPC_turning_cmd"],


    )
    train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False))

train_loader_windowed_data = []

for i in train_loader:

   for  x,y in i:

       train_loader_windowed_data.append(windowed_dataset_v2(x,y,input_window = 5, output_window = 1 , stride = 1, num_features = 10))
       pass


Xtrain  = train_loader_windowed_data[0][0]
Ytrain  = train_loader_windowed_data[0][1]
#print(Ytrain)

m1 = lstm_encoder_decoder_origin.lstm_seq2seq(input_size = 10, hidden_size = 100)
st= torch.load("./predict_model.pt")
m1.load_state_dict(st)
m1.eval()


with open("./test_accurancy.csv", 'w', newline='') as post_trials:
        writer = csv.writer(post_trials)
        writer.writerow(
        ["NPC_speed","location_X","location_Y","location_Z",
         "predict_NPC_speed","predict_NPC_location_X","predict_NPC_location_Y","predict_NPC_location_Z"
        ]
        )
        predict_speed = []
        ground_truth_speed =[]
        predict_location_x = []
        predict_location_y = []
        predict_location_z = []
        ground_truth_location_x = []
        ground_truth_location_y = []
        ground_truth_location_z = []
        speed_l2norm =[]
        predict_result= []

        for i in range(25):
            t1 = train_loader_windowed_data[0][0][0][i:i+5]
            t1 = torch.tensor(t1)
            t1 = t1.float()

            #print("current predict result is")
            #print(train_loader_windowed_data[0][1][0][i])
            ground_truth_speed.append(train_loader_windowed_data[0][1][0][i][0])
            ground_truth_location_x.append(train_loader_windowed_data[0][1][0][i][1])
            ground_truth_location_y.append(train_loader_windowed_data[0][1][0][i][2])
            ground_truth_location_z.append(train_loader_windowed_data[0][1][0][i][3])
            #print(ground_truth_location_x)
            #print("current ground truth is ")
            #print(t1)
            print("current prediction is ")
            y = m1.predict(t1, 1)
            print(y)




            #writer.writerow([train_loader_windowed_data[0][1][0][i][0],train_loader_windowed_data[0][1][0][i][1],train_loader_windowed_data[0][1][0][i][2],train_loader_windowed_data[0][1][0][i][3], y[0][0],y[0][1],y[0][2],y[0][3]])





ground_truth  = torch.from_numpy(train_loader_windowed_data[0][1][0]).type(torch.Tensor)
#print(ground_truth)
predict_result  = torch.from_numpy(np.asarray(predict_result)).type(torch.Tensor)
#print(predict_result)


predict_speed = torch.from_numpy(np.asarray(predict_speed)).type(torch.Tensor)
predict_location_x = torch.from_numpy(np.asarray(predict_location_x)).type(torch.Tensor)
predict_location_y = torch.from_numpy(np.asarray(predict_location_y)).type(torch.Tensor)
predict_location_z = torch.from_numpy(np.asarray(predict_location_z)).type(torch.Tensor)

ground_truth_speed = torch.from_numpy(np.asarray(ground_truth_speed)).type(torch.Tensor)
ground_truth_location_x = torch.from_numpy(np.asarray(ground_truth_location_x)).type(torch.Tensor)
ground_truth_location_y = torch.from_numpy(np.asarray(ground_truth_location_y)).type(torch.Tensor)
ground_truth_location_z = torch.from_numpy(np.asarray(ground_truth_location_z)).type(torch.Tensor)



"""
r2score = R2Score()
r2 = r2_score(ground_truth_speed, predict_speed)
print(" speed evaluation ")
print(ground_truth_speed)
print(predict_speed)
print(r2)
r2 = r2_score(ground_truth_location_x, predict_location_x)

print(" location x evaluation ")
print(ground_truth_location_x)
print(predict_location_x)
print(r2)
#r2 = r2_score(ground_truth_location_y, predict_location_y)
#print(r2)
#print(" location x evaluation ")
#r2 = r2_score(ground_truth_location_z, predict_location_z)
#print(r2)
#print(ground_truth_location_z)
#print(predict_location_z)
"""
