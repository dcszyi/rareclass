'''
Utilities for Loading data.
The input data file follows the same input for LibFM: http://www.libfm.org/libfm-1.42.manual.pdf

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@references:
'''
import numpy as np
import os

class LoadData(object):
    

    # Three files are needed in the path
    def __init__(self, loss_type, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
        self.features_M = X_train.shape[1]
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data_new(loss_type, X_train, Y_train, self.create_y_logloss(Y_train), 
                                            X_validation, Y_validation, self.create_y_logloss(Y_validation), 
                                            X_test, Y_test, self.create_y_logloss(Y_test)) 
        num_variable = self.truncate_features()
    
    
    def construct_data_new(self, loss_type, X_train, Y_train, Y_for_logloss_train, X_validation, Y_validation, Y_for_logloss_validation, X_test, Y_test, Y_for_logloss_test): 
        
        X_, Y_ , Y_for_logloss= X_train, Y_train, Y_for_logloss_train
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        print("# of training:" , len(Y_))

        X_, Y_ , Y_for_logloss= X_validation, Y_validation, Y_for_logloss_validation
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        print("# of validation:", len(Y_))

        X_, Y_ , Y_for_logloss = X_test, Y_test, Y_for_logloss_test
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        print("# of test:", len(Y_))

        return Train_data,  Validation_data,  Test_data
   
    def create_y_logloss(self, Y_):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        #e.g., [5:1 6:1 10:1] --> "5:1":0 "6:1":1 "10:1":2 --> [0,1,2]
        #X_ is a list where each line is a data sample 
        
        Y_for_logloss = []
        for y in Y_:
            if y > 0:
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append(v)
        
      
        #print(Y_[:10])
        #print(Y_for_logloss[:10])
        return Y_for_logloss
 
    def construct_dataset(self, X_, Y_): 
        Data_Dic = {}
        X_lens = [ len(line) for line in X_] 
        indexs = np.argsort(X_lens) #sort the samples from least features to most features, for frappe, each sample comes with the same size of features, so the order should be the same 
        Data_Dic['Y'] = [ Y_[i] for i in indexs]
        Data_Dic['X'] = [ X_[i] for i in indexs] 
        return Data_Dic
    
    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in range(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])]) #choose the minimal feature length over all training samples
        # truncate train, validation and test (assume the minimal length occurs in the training data, what if test/validation has fewer variables?  no need to apply this check if we one-hot encode the whole dataset at begnning
        for i in range(len(self.Train_data['X'])):            
            if len(self.Train_data['X'][i]) > num_variable:
                print("truncate a sample in training")
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in range(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in range(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable
