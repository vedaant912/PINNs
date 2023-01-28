import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import tqdm
from sklearn.metrics import mean_squared_error,r2_score
import statistics
import math
import os
from pytorchtools import EarlyStopping

def torch_seed(seed=0):
    '''
    Setting all the necessary values to generate reproducible results after training. 
    Parameter : seed (seed value)
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def calculate_test_error(model, X_test, y_test):
    '''
    Calculating the test/validation error using mean squared error.
    Parameters : model - Trained model
                  X_test - Features from test set.
                  y_test - Output values from test set.
    '''

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        y_test_pred = model(X_test)
        y_pred_list.append(y_test_pred.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    mse = mean_squared_error(y_test, y_pred_list[0])
    r_square = r2_score(y_test, y_pred_list[0])

    return mse, r_square

def get_train_val_test_data(data, seed):
    '''
    Fetching the train, test, validation dataset using seed value.
    Random Split Method.
    Parameter : data - Consisting of all the data.
                seed - seed value which is going to be used to split the dataset in  
                        train, test, and validation set.
    '''

    data = np.asarray(data)
    np.random.shuffle(data)
    
    X = data[:,:-2]
    y = data[:,-2:]
    
    # Get the train, val, and test data
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2,random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5,random_state=seed)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_train_test_val_data(seed):
    '''
    Dataset splitting on the basis of smile combination. Each smile combination is either in train set or in test set. Not in
    both of them.
    Parameter : seed - seed value used to split the dataset. 
    '''

    train_data = pd.read_csv('/data/smile_combination_data/training_data_seed_'+str(seed)+'_smile_combination.csv')
    test_data = pd.read_csv('/data/smile_combination_data/test_data_seed_'+str(seed)+'_smile_combination.csv')
    val_data = pd.read_csv('/data/smile_combination_data/validation_data_seed_'+str(seed)+'_smile_combination.csv')
    
    train_data.drop(['Unnamed: 0'],inplace=True,axis=1)
    test_data.drop(['Unnamed: 0'],inplace=True,axis=1)
    val_data.drop(['Unnamed: 0'],inplace=True,axis=1)
    
    train_data.drop(['SMILES_1','SMILES_2'], inplace=True, axis=1)
    test_data.drop(['SMILES_1','SMILES_2'], inplace=True, axis=1)
    val_data.drop(['SMILES_1','SMILES_2'], inplace=True, axis=1)
    
    return train_data.to_numpy(), test_data.to_numpy(), val_data.to_numpy()

def get_data_for_pinn_training():
    '''
    Fetching entire synthetic dataset for training only the pinn part.
    '''

    data = pd.read_csv('/data/pinn_synthetic_data/entire_dataset.csv')
    data.drop(['Unnamed: 0.1'], inplace=True, axis=1)
    data.drop(['Unnamed: 0'],inplace=True, axis=1)
    data.drop(['SMILES_1','SMILES_2'],inplace=True, axis=1)
    data = np.asarray(data)
    np.random.shuffle(data)

    X = data[:, :-2]
    y = data[:, -2:]

    return X, y


class Model2(nn.Module):
    ''' 
    Model with 5 layers. 
    '''

    def __init__(self, input_dim):
        super(Model2, self).__init__()
        self.layer1 = nn.Linear(input_dim, 4096)
        self.layer2 = nn.Linear(4096, 1024)
        self.layer3 = nn.Linear(1024, 256)
        self.layer4 = nn.Linear(256, 64)
        self.layer5 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x

def training(train_data, test_data, val_data, beta, bs, learning_rate, early_stopping_flag, pinn_only):
    '''
    Training the model using the train set and validation set. 
    Parameters : train_data - Training set
                 test_data - Test set
                 val_data - Validation - Set
                 beta - Weight used along with PINN loss
                 bs - Batch Size
                 learning_rate - Step Size
                 early_stopping_flag - Boolean. if True - Early Stopping else Training run for 'n' epochs
                 pinn_only - Boolean. if True - Only pinn loss is considered else neural network + pinn loss 
    '''
    
    X_train = train_data[:,:-2]
    y_train = train_data[:,-2:]
    X_test = test_data[:,:-2]
    y_test = test_data[:,-2:]
    X_val = val_data[:,:-2]
    y_val = val_data[:,-2:]
    
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).float()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).float()
    X_val = Variable(torch.from_numpy(X_val)).float()
    y_val = Variable(torch.from_numpy(y_val)).float()
    
    model_PINN_5 = Model2(X_train.shape[1])
    optimizer = torch.optim.Adam(model_PINN_5.parameters(), lr=learning_rate)
    loss_fn   = nn.MSELoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    EPOCHS = 5000
    batch_size = bs

    num_of_batches = math.ceil(X_train.size()[0] / batch_size)

    PINN_train_loss = np.zeros((EPOCHS,))
    PINN_validation_loss = np.zeros((EPOCHS,))
    PINN_loss1 = np.zeros((EPOCHS,))
    PINN_loss2 = np.zeros((EPOCHS,))
    PINN_val_loss1 = np.zeros((EPOCHS,))
    PINN_val_loss2 = np.zeros((EPOCHS,))
    mse = np.zeros((EPOCHS,))
    r_square = np.zeros((EPOCHS,))

    model_PINN_5 = model_PINN_5.to(device)
    
    if early_stopping_flag:
        patience = 50
        early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Get data for PINN part of the training
    seed_for_pinn_data = 20
    X_pinn, y_pinn = get_data_for_pinn_training(seed_for_pinn_data)
    X_pinn = torch.from_numpy(X_pinn).float()
    X_pinn = X_pinn.requires_grad_(True)
    X_pinn = X_pinn.to(device)
    batch_size_pinn = 2985

    for epoch in tqdm.trange(EPOCHS, position=0, leave=True):

        model_PINN_5.train()

        permutation = torch.randperm(X_train.size()[0])
        permutation_2  = torch.randperm(X_pinn.size()[0])
        j=0

        train_loss = 0
        validation_loss = 0
        total_loss1 = 0
        total_loss2 = 0

        for i in range(0,X_train.size()[0], batch_size):

            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_x.requires_grad_(True)
            batch_y = batch_y.to(device)

            # Compute data loss
            y_pred = model_PINN_5(batch_x)
            loss1 = loss_fn(y_pred, batch_y)
            
            # Compute the PINN loss
            indices_2 = permutation_2[j:j+batch_size_pinn]
            X_pinn_batch = X_pinn[indices_2]
            
            X_pinn_batch = X_pinn_batch.requires_grad_(True)
            x1 = X_pinn_batch[:, -3]
            x2 = X_pinn_batch[:, -2]
            y_pred_pinn = model_PINN_5(X_pinn_batch)

            # Differentiating the output with respect to all the inputs
            d_lngamma1_x1 = torch.autograd.grad(y_pred_pinn[:,0], X_pinn_batch, torch.ones_like(y_pred_pinn[:,0]), create_graph=True)[0]
            d_lngamma2_x2 = torch.autograd.grad(y_pred_pinn[:,1], X_pinn_batch, torch.ones_like(y_pred_pinn[:,1]), create_graph=True)[0]
            physics = x1*d_lngamma1_x1[:,-3] - x2*d_lngamma2_x2[:,-2]
            
            # LOSS 2
            loss2 = beta*torch.mean(physics**2)

            j = j + batch_size_pinn

            if pinn_only:
                loss = loss2
            else:
                loss = loss1 + loss2 # add the two loss terms together.
            
            # backpropogate joint losses.
            loss.backward()
            optimizer.step()

            # Add the losses to running loss.
            train_loss += loss.item() 

            total_loss1 += loss1
            total_loss2 += loss2


        X_val = X_val.to(device)
        X_val.requires_grad_(True)
        
        y_val = y_val.to(device)
        
        x_1 = X_val[:,-3]
        x_2 = X_val[:,-2]
        
        y_pred_val = model_PINN_5(X_val)
        validation_loss = loss_fn(y_pred_val, y_val)
        
        d_x1 = torch.autograd.grad(y_pred_val[:,0], X_val, torch.ones_like(y_pred_val[:,0]), create_graph=True)[0]
        d_x2 = torch.autograd.grad(y_pred_val[:,1], X_val, torch.ones_like(y_pred_val[:,1]), create_graph=True)[0]
        yph = x_1*d_x1[:,-3] - x_2*d_x2[:,-2]
        
        validation_loss_2 = beta*torch.mean(yph**2)	

        PINN_train_loss[epoch] = train_loss / batch_size
        PINN_validation_loss[epoch] = validation_loss + validation_loss_2
        PINN_val_loss1[epoch] = validation_loss
        PINN_val_loss2[epoch] = validation_loss_2

        PINN_loss1[epoch] = total_loss1 / batch_size
        PINN_loss2[epoch] = total_loss2 / batch_size
        
        mse[epoch], r_square[epoch] = calculate_test_error(model_PINN_5, X_val, y_val.cpu().numpy())

        if early_stopping_flag:   
            early_stopping(validation_loss, model_PINN_5)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
    if early_stopping_flag:
        model_PINN_5.load_state_dict(torch.load('checkpoint.pt'))
    
    return  PINN_train_loss, PINN_validation_loss, PINN_loss1, PINN_loss2, PINN_val_loss1, PINN_val_loss2, model_PINN_5, mse, r_square


def save_data(seed, early_stopping, train_loss, validation_loss, PINN_loss1, PINN_loss2, PINN_val_loss1, PINN_val_loss2, model_PINN_5, mse, r_square):
    '''
    Saving the data. 
    Parameters: seed - Used to save the files with seed included in the file name.
                early_stopping - Boolean, Append 'early_stopping' in file name if early_stopping is true.
                train_loss - loss values stored in an array on the basis of training data.
                validation_loss - loss values stored in an array on the basis of validation data.
                PINN_loss1 - Neural network loss on the basis of train data.
                PINN_loss2 - PINN loss generated on the basis of train data.
                PINN_val_loss1 - Neural network loss on the basis of validation data.
                PINN_val_loss2 - PINN loss generated on the basis of validation data.
                model_PINN_5 - Model trained at the end of all the epochs.
                mse - Mean Squared Error calculated on the basis of validation data.
                r_square - R Squared value measured on the basis of validation data.
    '''

    # If early_stopping is true, set text value to 'early_stopping'. variable text will be appended to the file name.
    if early_stopping:
        text = 'early_stopping'
    else:
        text = '5000_Epochs'

    # Saving all the data.
    np.savetxt('./train_loss_pinn_'+ str(seed) +'_' + text + '.csv',train_loss)
    np.savetxt('./val_loss_pinn_'+ str(seed) +'_' + text + '.csv',validation_loss)
    np.savetxt('./pinn_loss1_'+ str(seed) +'_' + text + '.csv',PINN_loss1)
    np.savetxt('./pinn_loss2_'+ str(seed) +'_' + text + '.csv',PINN_loss2)
    np.savetxt('./pinn_val_loss1_'+ str(seed) +'_' + text + '.csv',PINN_val_loss1)
    np.savetxt('./pinn_val_loss2_'+ str(seed) +'_' + text + '.csv',PINN_val_loss2)
    torch.save(model_PINN_5,'./model_pinn'+ str(seed) +'_' + text + '.pt')
    np.savetxt('./pinn_val_mse_'+ str(seed) +'_' + text + '.csv',mse)
    np.savetxt('./pinn_val_r_square_'+ str(seed) +'_' + text + '.csv',r_square)

# Seed value used for getting reproducible results.
seed_weights = 100

# Seed value used for fetching data.
seed_data = 20

# Flags for early stopping and pinn_only
early_stopping_flag = True
pinn_only = False
random_split = False

# Set the seed which will initialize the weights and help getting the same results.
torch_seed(seed_weights)

# Get training data on the basis of seed. Data based on Random split.
if random_split:
    data = pd.read_csv('/data/dataset/flatten_embedded_data.csv')
    data.drop(['Unnamed: 0'],inplace=True, axis=1)
    data.drop(['SMILES_1','SMILES_2'],inplace=True, axis=1)

    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(data,seed_data)
    train_loss, validation_loss, PINN_loss1, PINN_loss2, PINN_val_loss1, PINN_val_loss2, model_PINN_5, mse, r_square = training(X_train, y_train, X_val, y_val, X_test, y_test, 0.0003425, 2048, 0.001, early_stopping_flag)

else:
    # Get training data on the basis of seed. Data based on Fabian split.
    train_data, test_data, val_data = get_train_test_val_data(seed_data)
    train_loss, validation_loss, PINN_loss1, PINN_loss2, PINN_val_loss1, PINN_val_loss2, model_PINN_5, mse, r_square = training(train_data, test_data,val_data, 0.0003425, 2048, 0.001, early_stopping_flag, pinn_only)

save_data(seed_data, early_stopping_flag, train_loss, validation_loss, PINN_loss1, PINN_loss2, PINN_val_loss1, PINN_val_loss2, model_PINN_5, mse, r_square )
