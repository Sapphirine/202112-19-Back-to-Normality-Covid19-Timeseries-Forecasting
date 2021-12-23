
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class myRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,seq_len, dropout, bidirectional = False, type = 'lstm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        if type == 'lstm':
           self.rnn = nn.LSTM(input_size,hidden_size,num_layers,dropout = dropout, batch_first = True, bidirectional = bidirectional)
        elif type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        else:
          self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)



        if bidirectional:
            self.fc = nn.Linear(hidden_size*2,output_size)
        else:
            self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,X):   
        data  = X
        if self.bidirectional:
            output = self.rnn(data)[0]
            h_forward = output[:,-1,:self.hidden_size]
            h_backward = output[:,0,self.hidden_size:]
            output = torch.cat([h_forward,h_backward],dim = 1)
        else:
            output = self.rnn(data)[0][:,-1,:]
        output = self.fc(output)
        return output 

def compute_prediction(model,X_test):
  y_hat = model(X_test)
  return y_hat


def time_series_pred(model, last_day):

  predict = np.zeros([20,])
  for i in range(20):
    last_day = last_day.reshape(1,last_day.shape[0],1)
    last_day = torch.as_tensor(last_day).float().to('cuda')
    y_hat = compute_prediction(model,last_day)
    y_hat = y_hat.cpu().detach().numpy().flatten()
    last_day = last_day.cpu().numpy().flatten()
    last_day = np.delete(last_day,0) 
    last_day = np.concatenate([last_day,y_hat])
    predict[i] = y_hat[0]
  return predict


if __name__ == "__main__":

    df = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv')
    cases = df.loc[:,'CASE_COUNT']
    cases = np.asarray(cases)
    max = np.max(cases,axis = 0)
    new_20 = cases[-20:]
    new_20 = new_20/max
    model = torch.load('{}.pt'.format('rnn_minmax'))
    predict = time_series_pred(model,new_20)
    predict = predict*max
    file  = open('case_count_rnn.txt','w')
    for i in predict:
        file.write(str(int(i))+'\n')
    file.close()   

    