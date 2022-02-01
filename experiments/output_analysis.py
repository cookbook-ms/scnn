#!/usr/bin/env python3.8.5

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')
import numpy.linalg as la
import scnn.scnn
import scnn.chebyshev
import pandas as pd

num_realizations = 10

def main():
    prefix = './data/s2_3_collaboration_complex'
    percentage_missing_values=10    
    logdir = './experiments/output_%d' %(percentage_missing_values)
    #logdir = './experiments/output_order1'
    num_iter = 1000
    K = 5
    K1 = 2
    K2 = 3
    ## L1 loss
    training_loss1 = np.zeros((num_iter, num_realizations))
    test_loss1 = np.zeros((num_iter, num_realizations))
    training_loss2 = np.zeros((num_iter, num_realizations))
    test_loss2 = np.zeros((num_iter, num_realizations))
    for m in range(0,num_realizations):

        t1 = np.loadtxt("%s/loss_1_%sfilter_%d_%d.txt" %(logdir,percentage_missing_values,K,m))
        training_loss1[:,m]=t1[:,1]
        t2 =np.loadtxt("%s/loss_1_%sfilter_%d_%d_%d.txt" %(logdir,percentage_missing_values,K1,K2,m))
        training_loss2[:,m]= t2[:,1]
        t3 =np.loadtxt(("%s/testloss1_%d.txt" %(logdir,m)))
        test_loss1[:,m]=t3[:,1]
        t4=np.loadtxt(("%s/testloss2_%d.txt" %(logdir,m)))
        test_loss2[:,m]=t4[:,1]

        #print(training_loss1[:,1])

    fig1, ax = plt.subplots()    
    ax.semilogy(training_loss1.mean(1),label='SNN')
    ax.semilogy(training_loss2.mean(1),label='SCNN')
    ax.legend(fontsize=20)
    ax.set_xlabel("number of training", fontsize=20)
    ax.set_ylabel("L1 loss", fontsize=20)
    ax.set_title('training loss', fontsize=16)


    fig2, ax = plt.subplots()
    ax.semilogy(test_loss1.mean(1),label='SNN')
    ax.semilogy(test_loss2.mean(1),label='SCNN')
    ax.legend(fontsize=16)
    ax.set_xlabel("number of training", fontsize=16)
    ax.set_ylabel("L1 loss", fontsize=16)
    ax.set_title('test loss', fontsize=16)
    plt.show() 
 


        # ## NMSE
        # actual_data = []
        # # training true data
        # for d in range(0,topdim+1):
        #     actual_data.extend(np.loadtxt("%s/actual_%d.txt" %(logdir, d)))
        #     print(len(np.loadtxt("%s/actual_%d.txt" %(logdir, d))))
        
        # actual_all_data = []
        # # test all true data
        # for d in range(0,topdim+1):
        #     actual_all_data.extend(np.loadtxt("%s/actual_%d_all.txt" %(logdir, d)))
        #     print(len(np.loadtxt("%s/actual_%d_all.txt" %(logdir, d))))

        # # loss1 = np.loadtxt("%s/MSEloss_1_%dfilter_%d.txt" %(logdir,percentage_missing_values,K))
        # # loss2 = np.loadtxt("%s/MSEloss_1_%dfilter_%d_%d.txt" %(logdir,percentage_missing_values,K1,K2))     
        # # mse1 = loss1[:,1]
        # # mse2 = loss2[:,1]

        # # mse1 /= la.norm(actual_data) ** 2
        # # mse2 /= la.norm(actual_data) ** 2

        # # first compute and write the training and test loss into text files
        # training_loss1 = np.zeros((num_iter,1))
        # training_loss2 = np.zeros((num_iter,1))
        # test_loss1 = np.zeros((num_iter,1))
        # test_loss2 = np.zeros((num_iter,1))
        # # training_loss1 = []
        # # training_loss2 = []
        # # test_loss1 = []
        # # test_loss2 = []
        # f1 = open("%s/training_loss1_%d.txt" %(logdir,m), "w") 
        # f2 = open("%s/training_loss2_%d.txt" %(logdir,m), "w") 
        # f3 = open("%s/test_loss1_%d.txt" %(logdir,m), "w") 
        # f4 = open("%s/test_loss2_%d.txt" %(logdir,m), "w") 
        # for i in range(0,num_iter):
        #     training_output1 = []
        #     training_output2 = []
        #     test_output1 = []
        #     test_output2 = []
        #     for d in range(0,topdim+1):
        #         training_output1.extend(np.loadtxt("%s/MSEprediction1_%d_%d_%d.txt" %(logdir, i, d,m)))
        #         training_output2.extend(np.loadtxt("%s/MSEprediction2_%d_%d_%d.txt" %(logdir, i, d,m)))
        #         test_output1.extend(np.loadtxt("%s/MSEoutput1_%d_%d_%d.txt" %(logdir, i, d,m)))
        #         test_output2.extend(np.loadtxt("%s/MSEoutput2_%d_%d_%d.txt" %(logdir, i, d,m)))
        #     # print(len(actual_data), len(training_output1))
        #     # print(len(actual_all_data), len(test_output1))
        #     # print(np.subtract(training_output1,actual_data))
        
        #     # training_loss2.extend(la.norm(np.subtract(training_output2,actual_data)) ** 2 / la.norm(actual_data) ** 2)
        #     # test_loss1.extend(la.norm(np.subtract(test_output1,actual_all_data)) ** 2 / la.norm(actual_all_data) ** 2)
        #     # test_loss2.extend(la.norm(np.subtract(test_output2,actual_all_data)) ** 2 / la.norm(actual_all_data) ** 2)
        #     training_loss1[i] = (compare_nrmse(np.array(training_output1),np.array(actual_data)))
        #     training_loss2[i] = (compare_nrmse(np.array(training_output2),np.array(actual_data)))
        #     test_loss1[i] = (compare_nrmse(np.array(test_output1),np.array(actual_all_data)))
        #     test_loss2[i] = (compare_nrmse(np.array(test_output2),np.array(actual_all_data)))
        #         # np.savetxt("%s/training_loss1_%d.txt" %(logdir, i, d), detached_ys1[d][0,0,:])
        #         # np.savetxt("%s/training_loss2_%d.txt" %(logdir, i, d), detached_ys2[d][0,0,:])          
        #     f1.write("%d %f\n" %(i,training_loss1[i]))
        #     f1.flush()
        #     f2.write("%d %f\n" %(i,training_loss2[i]))
        #     f2.flush()
        #     f3.write("%d %f\n" %(i,test_loss1[i]))
        #     f3.flush()
        #     f4.write("%d %f\n" %(i,test_loss2[i]))
        #     f4.flush()

        
        # # boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)
        # # Ds=[(boundaries[i]) for i in range(topdim+1)]
        # # print(Ds[1])
        # # df = pd.DataFrame(Ds[1].todense())
        # # df.to_csv('boundaries_B2.csv', index=False)
        
        # # For Elvin
        # # signal = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
        # # raw_data=[(signal[i].values()) for i in range(len(signal))]
        # # print(len(raw_data[2]))
        # # df = pd.DataFrame(raw_data[2])
        # # df.to_csv('cochain_2.csv', index=False)



if __name__ == "__main__":
    main()



