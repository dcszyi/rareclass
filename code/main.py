# -*- coding:utf-8 -*-
import time
import datetime
import numpy as np
import argparse
import sys
import random
import os
import copy
import math  
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import tensorflow as tf 

from LoadData import LoadData
from NeuralFM import NeuralFM

  
  
def feature_extraction_NFM(X, y_binary, i_Pos_sample_set, i_Neg_sample_set, args):

    #the labeled data will come from both pos and neg 
    X_label = [X[i] for i in set.union(i_Pos_sample_set, i_Neg_sample_set)]
    y_label = [y_binary[i] for i in set.union(i_Pos_sample_set, i_Neg_sample_set)]
   
    X_train = np.asarray(X_label)
    Y_train = np.asarray(y_label)
    X_validation = np.asarray(X_label)
    Y_validation = np.asarray(y_label)
    X_test = copy.deepcopy(X) #set the whole dataset for test, so that we will get the new features 
    Y_test = copy.deepcopy(y_binary)
    
    data = LoadData(args.loss_type, X_train, Y_train, X_validation, Y_validation, X_test, Y_test) 
          
    # Training   
    model = NeuralFM(data.features_M, args.hidden_factor, args.layers, args.loss_type, 
                    args.pretrain, args.epoch, args.batch_size, args.lr, 
                    args.lamda, args.keep_prob, args.optimizer, 
                    args.batch_norm, args.activation_function, 
                    args.verbose, args.early_stop)
    model.train(data.Train_data, data.Validation_data, data.Test_data)
        
    features = model.get_deep_feature(data.Test_data)#model.get_bi_feature(data.Test_data)#
    return features 
    
    
def reconstruct_collaborative_binaryclass(X, y, y_multiclass, args):  
    
    
    #initialize the unlabeled dataset, only contain the index of all elements in X for quick reference later
    i_Interesting_set = set()
    i_Interesting_set.update([i for i in range(0,len(y))])
    #print("Initialize the unlabed set, # = %d" % len(i_Interesting_set))

    #choose the first sample, now we pick up the $first$ sample in that class in the whole dataset
    indices = [i_value for i_value, value in enumerate(y) if value == 1]    
    i_first_sample  = indices[0]
    i_Interesting_set.remove(i_first_sample)
    #print("Update the unlabeled set, # = %d" % len(i_Interesting_set))

    #initilize the labeled dataset (index set) and update the unlabeled dataset 
    i_Pos_sample_set = set()
    i_Pos_sample_set.add(i_first_sample)
    #print("Initialize positive set, # = %d" % len(i_Pos_sample_set))

    i_Neg_sample_set = set()
    i_Neg_random_first_round =  random.sample(range(len(i_Interesting_set)), 10) #Warning:Avoid np.random.randint(len(i_Interesting_set), size=100) as it may sample repeated elements 
    i_Neg_sample_set.update(i_Neg_random_first_round) #randomly select 100 samples from X as negative in the first round, note that we will reset the negaitve set once we have received the real negative samples, usually from the second round
    

    i_Interesting_set.difference_update(i_Neg_random_first_round) 
    #print("Update the unlabed set, # = %d" % len(i_Interesting_set))  

     
    
    print("[Initialize:]#Pos = %d, #Neg = %d, #Unlabel=%d, #Total = %d" %(len(i_Pos_sample_set), len(i_Neg_sample_set), len(i_Interesting_set), len(y)))
    
     
    print("[Iteration starts]") 
    
    construct_num = 100 
    reset_Neg_flag = 1    #once got the first negative labeled sample, the flag will be converted to false (0)  
    
    with open(args.output, 'w') as myfile: 
        myfile.write('index, #input-pos, #input-neg, real_class, index, #out-pos(result)\n') 

    
    for iter in range(construct_num): 
        if len(i_Pos_sample_set) >= sum(y): #no more positive sample in the unlabeled dataset
            print("Early Stop as all positives are found!")
            break
        print("**[Iteration %d / %d]" %(iter+1, construct_num))   
        
        candidate = list(i_Interesting_set) 
        similarity = dict()    
        
        
        if args.if_nfm: #feature adaptation through NFM, take the labeled set as training, the unlabled as testing (but we convert the whole dataset for easy-reference
            #print("NFM feature extraction....")
            X_nfm = feature_extraction_NFM(copy.deepcopy(X), y, i_Pos_sample_set, i_Neg_sample_set, args)      
        else: #
            #print("No NFM....")
            X_nfm = copy.deepcopy(X)
        
        #print(X_nfm.shape)
         
        for k in candidate: #for each unlabeled k, compared with labeled (pos + neg) and return a score for each k        
            sample_pos = X_nfm[[i_first_sample]]

            for i in i_Pos_sample_set:
                if i != i_first_sample    :
                    sample_pos = np.concatenate([sample_pos, X_nfm[[i]]], axis = 0)
                   
            #Similarity with postive samples 
            D = np.concatenate([sample_pos, -X_nfm[[k]]], axis=0).T
            e = np.ones(D.shape[1]-1) 
            gamma = 1 #Penalty
            B = np.eye(D.shape[1])
            B *= gamma
            u = np.expand_dims(np.concatenate([e, np.zeros(1)], axis = 0),axis=1)
            P = np.matmul(D.T,D) + B
            P_I = np.linalg.inv(P)
            x = np.matmul(P_I,u)/(np.matmul(np.matmul(u.T,P_I),u)) 
            a = x[0:-1] 
            b = x[-1]          
            similarity[k] = -1 * np.linalg.norm(sample_pos*a - X_nfm[k]) 
            
            #Similarity with negative samples 
            sample_neg = []
            if len(i_Neg_sample_set):
                Neg_sample = list(i_Neg_sample_set)
                sample_neg = X_nfm[[Neg_sample[0]]]
                for i in Neg_sample:
                    if i != Neg_sample[0]:
                        sample_neg = np.concatenate([sample_neg, X_nfm[[i]]], axis = 0)

                D = np.concatenate([sample_neg, -X_nfm[[k]]], axis=0).T
                e = np.ones(D.shape[1] - 1)
                gamma = 1  # Penalty
                B = np.eye(D.shape[1])
                B *= gamma
                u = np.expand_dims(np.concatenate([e, np.zeros(1)], axis=0), axis=1)
                P = np.matmul(D.T, D) + B
                P_I = np.linalg.inv(P)
                x = np.matmul(P_I, u) / (np.matmul(np.matmul(u.T, P_I), u))
                a = x[0:-1]
                b = x[-1]
                similarity[k] += np.linalg.norm(sample_neg*a - X_nfm[k]) 
        
            
        #sort all similairy scores from highest to lowest, check is the index, return the top-1         
        check = sorted(similarity.items(), key=lambda similarity: similarity[1], reverse = True)[0][0]
        
        print("[#positive = %d, #negative = %d] | Return class = %d (index = %d), reset = %d" % (len(sample_pos), len(sample_neg), y_multiclass[check], check, reset_Neg_flag))
        
        
        if y[check] == 1:
            i_Pos_sample_set.add(check)     
        else:
            if reset_Neg_flag: #since the Neg_set in the first round is randomly selected, so it should be updated once true negative feedback is received. 
                #print("reset the nagative set to true nagative")
                assert len(i_Neg_sample_set)==len(i_Neg_random_first_round), 'reset nagative set error'
                i_Interesting_set.update(i_Neg_sample_set) #move the first round values to unlabeled set
                i_Neg_sample_set = set()
                
            i_Neg_sample_set.add(check)
            reset_Neg_flag = 0  
        
        i_Interesting_set.remove(check)    
        
        with open(args.output, 'a') as myfile: 
              myfile.write("%d, %d, %d, %d, %d, %d\n"%(iter, 
                                len(sample_pos), len(sample_neg), 
                                y_multiclass[check], check, len(i_Pos_sample_set)))  


       
         
  
if __name__ == '__main__':
    
   
    
    parser = argparse.ArgumentParser(description="Run Neural FM FOR RARE CLASS DETECTION.") 

    parser.add_argument('--datapath', default='../data/Pokerhand/X.onehot.npy', help='One-hot encoded data') #One-hot encoded for both categorical and numeric attributes
    parser.add_argument('--labelpath', default='../data/Pokerhand/y.binary_class0.npy', help='binary label') #binary label 0/1
    parser.add_argument('--if_nfm', type=int, default=0, help='if apply NFM feature learning')  
    parser.add_argument('--output', default='../output/Pokerhand.binary_class0.out', help='file to record the result') 
    parser.add_argument('--metapath', default='../data/Pokerhand/y.multiclass.npy', help='binary label') #optional, just for analysis purpose, if not exist, set as the same as labelpath
    args = parser.parse_args()
    t = datetime.datetime.now().strftime("%y%m%d%H%M")
    
    
    
    #set the NFM parameters
    if args.if_nfm:
        args.output = args.output + ".nfm"  
        args.verbose = 0
        args.batch_norm = 1
        args.batch_size = 128
        args.early_stop = 1 
        args.pretrain = 0
        args.epoch = 50
        args.lamda = 0 
        args.optimizer ='AdagradOptimizer'
        args.layers = [64] 
        args.activation_function = tf.nn.relu 
        args.loss_type = 'square_loss'
        args.hidden_factor = 64
        args.lr = 0.01
        args.keep_prob = [0.5,0.5]
    else:
        args.output = args.output + ".baseline"
    
    args.output = args.output + ".t" + str(t) # to distinguish for multiple trials
    
    X = np.load(args.datapath) 
    y = np.load(args.labelpath)  
    y_multiclass = np.load(args.metapath)
    
    #record the parameters
    with open(args.output+'.para', 'w') as f: 
         print(vars(args), file=f)
         print('\n')   

    reconstruct_collaborative_binaryclass(X, y, y_multiclass, args)

 
    #output:
    #without feature learning: 
    #../output/Pokerhand.class_0.out.baseline.t19062001 
    #../output/Pokerhand.class_0.out.baseline.para
    #with NFM feature learning:
    #../output/Pokerhand.class_0.out.nfm.t19062205
    #../output/Pokerhand.class_0.out.nfm.t19062205.para
        
        
        
        
        
        
        
        
        