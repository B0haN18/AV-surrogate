import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

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

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 3):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''



        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))


        return lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers = 3,output_size=1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):

        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)
        #self.hideen_NPC = nn.Linear(16, 64)
        self.fc_npc = nn.Sequential(nn.Linear(16, 64), nn.Linear(64, 8), nn.Sigmoid())
        #self.hideen_AV = nn.Linear(16, 32)
        self.fc_av = nn.Sequential(nn.Linear(16, 64), nn.Linear(64, 6) , nn.ReLU())





    def train_model_with_fc(self, dataloader_l, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.01, dynamic_tf = False):
         # initialize array of losses
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()
        criterion2 = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()
        # calculate number of batch iterations
        with trange(n_epochs) as tr:
            for it in tr:
                batch_loss =0.

                batch_loss_npc = 0.
                batch_loss_av = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                index = 0
                for dataloader in dataloader_l:
                    #print(dataloader[0])

                    input_tensor = dataloader[0]

                    target_tensor = dataloader[1]
                    input_tensor, target_tensor, _, _ = numpy_to_torch(input_tensor, target_tensor, input_tensor, target_tensor)
                    n_batches = int(input_tensor.shape[1] / batch_size)
                    #print(n_batches)
                    #print(target_tensor)

                    for b in range(n_batches):
                        # select data
                        input_batch = input_tensor[:, b: b + batch_size, :]

                        target_batch = target_tensor[:, b: b + batch_size, :]
                        target_batch_npc = target_tensor[:, b: b + batch_size, 0:8]
                        #print(target_batch_npc)
                        target_batch_av = target_tensor[:, b: b + batch_size, 8:14]


                        # outputs tensor
                        outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                        # initialize hidden state
                        encoder_hidden = self.encoder.init_hidden(batch_size)
                        # zero the gradient
                        optimizer.zero_grad()
                        # encoder outputs
                        encoder_output, encoder_hidden = self.encoder(input_batch)

                        # decoder with teacher forcing
                        decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                        decoder_hidden = encoder_hidden
                        if training_prediction == 'recursive':
                            # predict recursively

                            for t in range(target_len):
                                if(t==0):
                                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                    outputs[t] = decoder_output
                                    decoder_input = decoder_output
                                    #print(type(target_batch))
                                    #output_hidden_npc = self.hideen_NPC(decoder_output)
                                    output_fc_npc = self.fc_npc(decoder_output)
                                    output_fc_L_npc =  output_fc_npc
                                    #output_hidden_av = self.hideen_NPC(decoder_output)
                                    output_fc_av = self.fc_av(decoder_output)
                                    output_fc_L_av =  output_fc_av


                                else:


                                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                    outputs[t] = decoder_output
                                    decoder_input = decoder_output
                                    #print(type(target_batch))
                                    #output_fc = self.fc(decoder_output)
                                    output_hidden_npc = self.hideen_NPC(decoder_output)
                                    output_fc_npc = self.fc_npc(output_hidden_npc)
                                    output_fc_L_npc = torch.cat([output_fc_L_npc,output_fc_npc])
                                    output_hidden_av = self.hideen_NPC(decoder_output)
                                    output_fc_av = self.fc_av(output_hidden_av)
                                    output_fc_L_av = torch.cat([output_fc_L_av,output_fc_av])
                                    #print(output_fc_L)


                        # compute the loss
                        #print("target_batch")
                        #print(target_batch[:,0].shape)
                        #print("prediction")
                        #print(output_fc_L)

                        loss_npc = criterion(output_fc_L_npc, target_batch_npc[:,0])
                        loss_av = criterion2(output_fc_L_av, target_batch_av[:,0])

                        #print(loss)
                        """
                        if(loss>1):
                            print(index)
                            print("target_batch")
                            print(target_batch[:,0])
                            print("prediction")
                            print(output_fc_L)
                        """
                        #batch_loss_npc += loss_npc.item()
                        #batch_loss_av += loss_av.item()

                        batch_loss +=  loss_npc.item() + loss_av.item()
                        loss = loss_npc+loss_av
                        # backpropagation
                        loss.backward()
                        #loss_npc.backward(retain_graph=True)
                        #loss_av.backward()
                        optimizer.step()
                    # loss for epoch
                    batch_loss /= n_batches
                    #batch_loss_npc /= n_batches
                    #batch_loss_av /= n_batches
                    losses[it] = batch_loss


                    # progress bar
                    tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    index  +=1

        return losses



    def predict(self, input_tensor, target_len):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])
        #output_hidden_npc = torch.zeros(target_len,  32)
        #output_hidden_av = torch.zeros(target_len,  32)
        output_fc_npc = torch.zeros(target_len,  8)
        output_fc_av = torch.zeros(target_len,  6)

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            #print(outputs[t])
            #adding a hidden layer
            #output_hidden_npc[t] = self.hideen_NPC(outputs[t])
            #output_hidden_av[t] = self.hideen_AV(outputs[t])

            #adding a fc layer predict
            output_fc_npc[t] = self.fc_npc(outputs[t])
            output_fc_av[t] = self.fc_av(outputs[t])
            decoder_input = decoder_output
        np_outputs_npc = output_fc_npc.detach().numpy()
        np_outputs_av = output_fc_av.detach().numpy()

        return np_outputs_npc,np_outputs_av
