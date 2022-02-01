#!/usr/bin/env python3.8.5

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')
import numpy.linalg as la
import pandas as pd
from numpy import savetxt
import math




def main():
    prefix = './data/s2_3_collaboration_complex'
    starting_node= 150250
    percentage_missing_values=[10,20,30,40,50]
    topdim = 5

    ind = 999 # the index of the last training results     
    epsilon = 0.05
    num_realizations = 10
    accuracy1 = np.zeros((num_realizations, topdim+1,len(percentage_missing_values)))
    accuracy2 = np.zeros((num_realizations, topdim+1,len(percentage_missing_values)))
    nmse1 = np.zeros((num_realizations, topdim+1,len(percentage_missing_values)))
    nmse2 = np.zeros((num_realizations, topdim+1,len(percentage_missing_values)))
    for i in range(0, len(percentage_missing_values)):
        logdir = './experiments/output_%d' %(percentage_missing_values[i])
        #logdir = './experiments/output_order1' #%(percentage_missing_values[i])
        actual_all_data = []
        for d in range(0,topdim+1):
            actual_all_data = np.loadtxt("%s/actual_%d_all.txt" %(logdir, d))
            for m in range(0,num_realizations):
                cochain_dim = len(actual_all_data)
                test_output1 = np.loadtxt("%s/output1_%d_%d_%d.txt" %(logdir, ind, d, m))
                test_output2 = np.loadtxt("%s/output2_%d_%d_%d.txt" %(logdir, ind, d, m))
                diff1 = np.abs(np.subtract(actual_all_data,test_output1)) 
                diff2 = np.abs(np.subtract(actual_all_data,test_output2)) 
                accuracy1[m,d,i] = np.sum(diff1<=np.abs(epsilon*actual_all_data))/cochain_dim
                accuracy2[m,d,i] = np.sum(diff2<=np.abs(epsilon*actual_all_data))/cochain_dim
                #nmse1[m,d,i]

    print(accuracy1.mean(0), "\n",accuracy2.mean(0))
    print(accuracy1.std(0), "\n",accuracy2.std(0))

    # np.savetxt('accuracy1_mean_order1.csv',np.round(accuracy1.mean(0),4),delimiter=',')
    # np.savetxt('accuracy2_mean_order1.csv',np.round(accuracy2.mean(0),4),delimiter=',')
    # np.savetxt('accuracy1_std_order1.csv',np.round(accuracy1.std(0),5),delimiter=',')
    # np.savetxt('accuracy2_std_order1.csv',np.round(accuracy2.std(0),5),delimiter=',')

    np.savetxt('accuracy1_mean.csv',np.round(accuracy1.mean(0),4),delimiter=',')
    np.savetxt('accuracy2_mean.csv',np.round(accuracy2.mean(0),4),delimiter=',')
    np.savetxt('accuracy1_std.csv',np.round(accuracy1.std(0),5),delimiter=',')
    np.savetxt('accuracy2_std.csv',np.round(accuracy2.std(0),5),delimiter=',')
    fig, ax = plt.subplots(1,6, figsize=(15,2),sharex=True, sharey=True)
    for d in range(0, topdim+1):
        # plot the mean
        ax[d].semilogy(percentage_missing_values, accuracy1[:,d,:].mean(axis=0), label='SCNN1')
        ax[d].semilogy(percentage_missing_values, accuracy2[:,d,:].mean(axis=0), label='SCNN2')
        # add the shadow for vairance
        ax[d].set_title('dim %d'%d)
        ax[d].legend()
        ax[d].set_xlim((10,50))
    fig.tight_layout()

    fig2, ax2 = plt.subplots(1,6, figsize=(15,2),sharex=True, sharey=True)
    for d in range(0, topdim+1):
        # plot the mean
        ax2[d].semilogy(percentage_missing_values, accuracy1[:,d,:].std(axis=0), label='SCNN1')
        ax2[d].semilogy(percentage_missing_values, accuracy2[:,d,:].std(axis=0), label='SCNN2')
        # add the shadow for vairance
        ax2[d].set_title('dim %d'%d)
        ax2[d].legend()
        ax2[d].set_xlim((10,50))
    fig2.tight_layout()
 
    plt.show()


        
        



if __name__ == "__main__":
    main()



