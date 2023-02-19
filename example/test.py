from MTO_SW_LSTM import MTO_SW_LSTM

window_size = 20
hidden_size = 100
num_layers = 2
n_features = 10
stride = 10
bsize = 10
device = 'cuda'
bidir = True
nout = [50, 100, 10]
dropout = 0.1
dropout2 = 0.2

rnn = MTO_SW_LSTM(window_size,hidden_size,num_layers,n_features,stride,bsize,device,bidir,nout,dropout,dropout2)
print(rnn)
import torch
train_data = []
val_data = []

for i in range(1):
    X_seq = torch.randn((500,n_features)).float().to(device)
    y_seq = torch.randn((1,nout[-1])).float().to(device)
    train_data.append((X_seq,y_seq))
    X_seq = torch.randn((500,n_features)).float().to(device)
    y_seq = torch.randn((1,nout[-1])).float().to(device)
    val_data.append((X_seq,y_seq))
print(X_seq)
print(y_seq)
