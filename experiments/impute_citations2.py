#!/usr/bin/env python3.8.5

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy.linalg as la
import numpy as np

import sys
sys.path.append('.')
import scnn.scnn
import scnn.chebyshev


# class MySCNN(nn.Module):
#     def __init__(self, colors = 1):
#         super().__init__()

#         assert(colors > 0)
#         self.colors = colors

#         num_filters = 30 #20
#         variance = 0.01 #0.001
#         K = 4
#         # Degree 0 convolutions.
#         self.C0_1 = scnn.scnn.SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
#         self.C0_2 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
#         self.C0_3 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)

#         # Degree 1 convolutions.
#         self.C1_1 = scnn.scnn.SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
#         self.C1_2 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
#         self.C1_3 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)

#         # Degree 2 convolutions.
#         self.C2_1 = scnn.scnn.SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
#         self.C2_2 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
#         self.C2_3 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)

#         # Degree 3 convolutions.
#         self.C3_1 = scnn.scnn.SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
#         self.C3_2 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
#         self.C3_3 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)

#         # Degree 4 convolutions.
#         self.C4_1 = scnn.scnn.SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
#         self.C4_2 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
#         self.C4_3 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)

#         # Degree 5 convolutions.
#         self.C5_1 = scnn.scnn.SimplicialConvolution(K, self.colors, num_filters*self.colors, variance=variance)
#         self.C5_2 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, num_filters*self.colors, variance=variance)
#         self.C5_3 = scnn.scnn.SimplicialConvolution(K, num_filters*self.colors, self.colors, variance=variance)



#     def forward(self, Ls, Ds, adDs, xs):
#         assert(len(xs) == 6) # The three degrees are fed together as a list.

#         assert(len(Ls) == len(Ds))
#         Ms = [L.shape[0] for L in Ls]
#         Ns = [D.shape[0] for D in Ds]

#         Bs = [x.shape[0] for x in xs]
#         C_ins = [x.shape[1] for x in xs]
#         Ms = [x.shape[2] for x in xs]

#         assert(Ms == [D.shape[1] for D in Ds])
#         assert(Ms == [L.shape[1] for L in Ls])
#         assert([adD.shape[0] for adD in adDs] == [D.shape[1] for D in Ds])
#         assert([adD.shape[1] for adD in adDs] == [D.shape[0] for D in Ds])

#         assert(Bs == len(Bs)*[Bs[0]])
#         assert(C_ins == len(C_ins)*[C_ins[0]])

#         out0_1 = self.C0_1(Ls[0], xs[0]) #+ self.D10_1(xs[1])
#         out1_1 = self.C1_1(Ls[1], xs[1]) #+ self.D01_1(xs[0]) + self.D21_1(xs[2])
#         out2_1 = self.C2_1(Ls[2], xs[2]) #+ self.D12_1(xs[1])
#         out3_1 = self.C3_1(Ls[3], xs[3]) #+ self.D12_1(xs[1])
#         out4_1 = self.C4_1(Ls[4], xs[4]) #+ self.D12_1(xs[1])
#         out5_1 = self.C5_1(Ls[5], xs[5]) #+ self.D12_1(xs[1])


#         out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1)) #+ self.D10_2(nn.LeakyReLU()(out1_1))
#         out1_2 = self.C1_2(Ls[1], nn.LeakyReLU()(out1_1)) #+ self.D01_2(nn.LeakyReLU()(out0_1)) + self.D21_2(nn.LeakyReLU()(out2_1))
#         out2_2 = self.C2_2(Ls[2], nn.LeakyReLU()(out2_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))
#         out3_2 = self.C3_2(Ls[3], nn.LeakyReLU()(out3_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))
#         out4_2 = self.C4_2(Ls[4], nn.LeakyReLU()(out4_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))
#         out5_2 = self.C5_2(Ls[5], nn.LeakyReLU()(out5_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))

#         out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2)) #+ self.D10_3(nn.LeakyReLU()(out1_2))
#         out1_3 = self.C1_3(Ls[1], nn.LeakyReLU()(out1_2)) #+ self.D01_3(nn.LeakyReLU()(out0_2)) + self.D21_2(nn.LeakyReLU()(out2_2))
#         out2_3 = self.C2_3(Ls[2], nn.LeakyReLU()(out2_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))
#         out3_3 = self.C3_3(Ls[3], nn.LeakyReLU()(out3_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))
#         out4_3 = self.C4_3(Ls[4], nn.LeakyReLU()(out4_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))
#         out5_3 = self.C5_3(Ls[5], nn.LeakyReLU()(out5_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))

#         #return [out0_3, torch.zeros_like(xs[1]), torch.zeros_like(xs[2])]
#         #return [torch.zeros_like(xs[0]), out1_3, torch.zeros_like(xs[2])]
#         return [out0_3, out1_3, out2_3, out3_3, out4_3, out5_3]

# ms's code
class MySCNN2(nn.Module):
    def __init__(self, colors = 1):
        super().__init__()

        assert(colors > 0)
        self.colors = colors

        num_filters = 30 #20
        variance = 0.01 #0.001
        K1 = 1#2
        K2 = 2#3
        # Degree 0 convolutions.
        self.C0_1 = scnn.scnn.SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C0_2 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C0_3 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.scnn.SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C1_2 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C1_3 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.scnn.SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C2_2 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C2_3 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C3_1 = scnn.scnn.SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C3_2 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C3_3 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C4_1 = scnn.scnn.SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C4_2 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C4_3 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C5_1 = scnn.scnn.SimplicialConvolution2(K1, K2, self.colors, num_filters*self.colors, variance=variance)
        self.C5_2 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C5_3 = scnn.scnn.SimplicialConvolution2(K1, K2, num_filters*self.colors, self.colors, variance=variance)


    def forward(self, Lls, Lus, Ds, adDs, xs):
        assert(len(xs) == 6) # The three degrees are fed together as a list.

        assert(len(Lls) == len(Ds))
        assert(len(Lus) == len(Ds))
        Ms = [Ll.shape[0] for Ll in Lls]
        Ms = [Lu.shape[0] for Lu in Lus]
        Ns = [D.shape[0] for D in Ds]

        Bs = [x.shape[0] for x in xs]
        C_ins = [x.shape[1] for x in xs]
        Ms = [x.shape[2] for x in xs]

        assert(Ms == [D.shape[1] for D in Ds])
        assert(Ms == [Ll.shape[1] for Ll in Lls])
        assert(Ms == [Lu.shape[1] for Lu in Lus])
        assert([adD.shape[0] for adD in adDs] == [D.shape[1] for D in Ds])
        assert([adD.shape[1] for adD in adDs] == [D.shape[0] for D in Ds])

        assert(Bs == len(Bs)*[Bs[0]])
        assert(C_ins == len(C_ins)*[C_ins[0]])

        out0_1 = self.C0_1(Lls[0], Lus[0], xs[0]) #+ self.D10_1(xs[1])
        out1_1 = self.C1_1(Lls[1], Lus[1], xs[1]) #+ self.D01_1(xs[0]) + self.D21_1(xs[2])
        out2_1 = self.C2_1(Lls[2], Lus[2], xs[2]) #+ self.D12_1(xs[1])
        out3_1 = self.C3_1(Lls[3], Lus[3], xs[3]) #+ self.D12_1(xs[1])
        out4_1 = self.C4_1(Lls[4], Lus[4], xs[4]) #+ self.D12_1(xs[1])
        out5_1 = self.C5_1(Lls[5], Lus[5], xs[5]) #+ self.D12_1(xs[1])

        out0_2 = self.C0_2(Lls[0], Lus[0], nn.LeakyReLU()(out0_1)) #+ self.D10_2(nn.LeakyReLU()(out1_1))
        out1_2 = self.C1_2(Lls[1], Lus[1], nn.LeakyReLU()(out1_1)) #+ self.D01_2(nn.LeakyReLU()(out0_1)) + self.D21_2(nn.LeakyReLU()(out2_1))
        out2_2 = self.C2_2(Lls[2], Lus[2], nn.LeakyReLU()(out2_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))
        out3_2 = self.C3_2(Lls[3], Lus[3], nn.LeakyReLU()(out3_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))
        out4_2 = self.C4_2(Lls[4], Lus[4], nn.LeakyReLU()(out4_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))
        out5_2 = self.C5_2(Lls[5], Lus[5], nn.LeakyReLU()(out5_1)) #+ self.D12_2(nn.LeakyReLU()(out1_1))


        out0_3 = self.C0_3(Lls[0], Lus[0], nn.LeakyReLU()(out0_2)) #+ self.D10_3(nn.LeakyReLU()(out1_2))
        out1_3 = self.C1_3(Lls[1], Lus[1], nn.LeakyReLU()(out1_2)) #+ self.D01_3(nn.LeakyReLU()(out0_2)) + self.D21_2(nn.LeakyReLU()(out2_2))
        out2_3 = self.C2_3(Lls[2], Lus[2], nn.LeakyReLU()(out2_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))
        out3_3 = self.C3_3(Lls[3], Lus[3], nn.LeakyReLU()(out3_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))
        out4_3 = self.C4_3(Lls[4], Lus[4], nn.LeakyReLU()(out4_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))
        out5_3 = self.C5_3(Lls[5], Lus[5], nn.LeakyReLU()(out5_2)) #+ self.D12_3(nn.LeakyReLU()(out1_2))


        #return [out0_3, torch.zeros_like(xs[1]), torch.zeros_like(xs[2])]
        #return [torch.zeros_like(xs[0]), out1_3, torch.zeros_like(xs[2])]
        return [out0_3, out1_3, out2_3, out3_3, out4_3, out5_3]

def main():
    torch.manual_seed(1337)
    np.random.seed(1337)

    prefix = sys.argv[1] ##input
    logdir = sys.argv[2] ##output
    starting_node=sys.argv[3]
    percentage_missing_values=sys.argv[4]
    cuda = False

    topdim = 5
    num_realizations = 10
    num_iter = 1000
    K1 = 2 # lower part order
    K2 = 3 # upper part order
    laplacians = np.load('{}/{}_laplacians.npy'.format(prefix,starting_node),allow_pickle=True)
    laplacians_down = np.load('{}/{}_laplacians_down.npy'.format(prefix,starting_node),allow_pickle=True)
    laplacians_up = np.load('{}/{}_laplacians_up.npy'.format(prefix,starting_node),allow_pickle=True)
    boundaries = np.load('{}/{}_boundaries.npy'.format(prefix,starting_node),allow_pickle=True)


    #Ls =[scnn.scnn.coo2tensor(scnn.chebyshev.normalize(laplacians[i],half_interval=True)) for i in range(topdim+1)] #####scnn.chebyshev.normalize 
    #Ls =[scnn.scnn.coo2tensor(laplacians[i]) for i in range(topdim+1)]
    Ls =[scnn.scnn.coo2tensor(scnn.chebyshev.normalize2(laplacians[i],laplacians[i],half_interval=True)) for i in range(topdim+1)]
    Ds=[scnn.scnn.coo2tensor(boundaries[i].transpose()) for i in range(topdim+1)]
    adDs=[scnn.scnn.coo2tensor(boundaries[i]) for i in range(topdim+1)]
    Lls=[scnn.scnn.coo2tensor(scnn.chebyshev.normalize2(laplacians[i],laplacians_down[i],half_interval=True)) for i in range(topdim+1)] 
    Lus=[scnn.scnn.coo2tensor(scnn.chebyshev.normalize2(laplacians[i],laplacians_up[i],half_interval=True)) for i in range(topdim+1)] 


    for m in range(num_realizations):
        # network1 = MySCNN(colors = 1)
        network2 = MySCNN2(colors = 1)

        learning_rate = 0.001
        # optimizer1 = torch.optim.Adam(network1.parameters(), lr=learning_rate)
        optimizer2 = torch.optim.Adam(network2.parameters(), lr=learning_rate)
        criterion = nn.L1Loss(reduction="sum")
        #criterion = nn.MSELoss(reduction="sum")

        batch_size = 1

        # num_params = 0
        # print("Parameter counts:")
        # for param in network1.parameters():
        #     p = np.array(param.shape, dtype=int).prod()
        #     print(p)
        #     num_params += p
        # print("Total number of parameters of SCNN1: %d" %(num_params))
        num_params = 0
        for param in network2.parameters():
            p = np.array(param.shape, dtype=int).prod()
            print(p)
            num_params += p
        print("Total number of parameters of SCNN2: %d" %(num_params))
        # this is the sampling mask 
        masks_all_deg = np.load('{}/{}_percentage_{}_known_values_{}.npy'.format(prefix,starting_node,percentage_missing_values,m),allow_pickle=True) ## positive mask= indices that we keep ##1 mask #entries 0 degree
        masks=[list(masks_all_deg[i].values()) for i in range(len(masks_all_deg))]
        
        # losslogf1 = open("%s/loss_1_%sfilter_%d_%d.txt" %(logdir,percentage_missing_values,K,m), "w")
        losslogf2 = open("%s/loss_1_%sfilter_%d_%d_%d.txt" %(logdir,percentage_missing_values,K1,K2,m), "w")
        # testlosslogf1 = open("%s/testloss1_%d.txt" %(logdir,m), "w") 
        testlosslogf2 = open("%s/testloss2_%d.txt" %(logdir,m), "w") 

        # the underlying true cochains
        cochain_target_alldegs = []
        signal = np.load('{}/{}_cochains.npy'.format(prefix,starting_node),allow_pickle=True)
        raw_data=[list(signal[i].values()) for i in range(len(signal))]
        for d in range(0, topdim+1):
            cochain_target = torch.zeros((batch_size, 1, len(raw_data[d])), dtype=torch.float, requires_grad = False)
            for i in range(0, batch_size):
                cochain_target[i, 0, :] = torch.tensor(raw_data[d], dtype=torch.float, requires_grad = False)

            cochain_target_alldegs.append(cochain_target)

        # the input cochains, in which the missing values are replaced by the median of the knowns     
        cochain_input_alldegs = []
        # here, we want to have multiple realizations, mutiple inputs to generate mean performance
        signal = np.load('{}/{}_percentage_{}_input_damaged_{}.npy'.format(prefix,starting_node,percentage_missing_values,m),allow_pickle=True)
        raw_data=[list(signal[i].values()) for i in range(len(signal))]
        # print(raw_data[0])
        for d in range(0, topdim+1):
            cochain_input = torch.zeros((batch_size, 1, len(raw_data[d])), dtype=torch.float, requires_grad = False)
            for i in range(0, batch_size):
                cochain_input[i, 0, :] = torch.tensor(raw_data[d], dtype=torch.float, requires_grad = False)
                # print(len(cochain_input[i, 0, :]))

            cochain_input_alldegs.append(cochain_input)
        
        # the percentage of the rest values
        print([float(len(masks[d]))/float(len(cochain_target_alldegs[d][0,0,:])) for d in range(0,topdim+1)])

        for d in range(0, topdim+1):
            actuallogf = open("%s/actual_%d.txt" %(logdir, d), "w") # only the sampled underlying true data
            #print(torch.eq(cochain_target_alldegs[d][0, 0, masks[d]] , cochain_input_alldegs[d][0,0,:]))
            for x in cochain_target_alldegs[d][0, 0, masks[d]]:
                actuallogf.write("%f " %(x))
            actuallogf.write("\n")
            actuallogf.close()

        for d in range(0, topdim+1):
            actuallogf_all = open("%s/actual_%d_all.txt" %(logdir, d), "w") # true data
            #print(torch.eq(cochain_target_alldegs[d][0, 0, masks[d]] , cochain_input_alldegs[d][0,0,:]))
            for x in cochain_target_alldegs[d][0, 0, :]:
                actuallogf_all.write("%f " %(x))
            actuallogf_all.write("\n")
            actuallogf_all.close()

        ## training process
        for i in range(0, num_iter):
            xs = [cochain_input.clone() for cochain_input in cochain_input_alldegs] # the input 
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # ys1 = network1(Ls, Ds, adDs, xs) # the full prediction
            ys2 = network2(Lls, Lus, Ds, adDs, xs)
            # training loss which are the discrepency between the sampled 
            # loss1 = torch.FloatTensor([0.0])
            loss2 = torch.FloatTensor([0.0])
            for b in range(0, batch_size):
                for d in range(0, topdim+1):
                    # loss1 += criterion(ys1[d][b, 0, masks[d]], cochain_target_alldegs[d][b, 0, masks[d]]) # compare the predicted ones with the underlying true ones only on the sampled ones
                    loss2 += criterion(ys2[d][b, 0, masks[d]], cochain_target_alldegs[d][b, 0, masks[d]]) 
            # detached_ys1 = [ys1[d].detach() for d in range(0, topdim+1)]
            detached_ys2 = [ys2[d].detach() for d in range(0, topdim+1)]

            # store the all predicted values in a txt file
            for d in range(0,topdim+1):
                # np.savetxt("%s/output1_%d_%d_%d.txt" %(logdir, i, d, m), detached_ys1[d][0,0,:])
                np.savetxt("%s/output2_%d_%d_%d.txt" %(logdir, i, d, m), detached_ys2[d][0,0,:])

            # only write the masked predicted ones
            for d in range(0, topdim+1):
                # predictionlogf1 = open("%s/prediction1_%d_%d_%d.txt" %(logdir, i, d, m), "w")
                predictionlogf2 = open("%s/prediction2_%d_%d_%d.txt" %(logdir, i, d, m), "w")
                #actuallogf = open("%s/actual_%d_%d.txt" %(logdir, i, d), "w") # the underlying true ones no need to write every echo
            #only write the masked ones
                for b in range(0, batch_size):
                    # write the predicted ones only on the masked ones
                    # for y in detached_ys1[d][b, 0, masks[d]]:
                    #     predictionlogf1.write("%f " %(y))
                    # predictionlogf1.write("\n")
                    for y in detached_ys2[d][b, 0, masks[d]]:
                        predictionlogf2.write("%f " %(y))
                    predictionlogf2.write("\n")
                    # write the underlying true values
    #                 for x in cochain_target_alldegs[d][b, 0, masks[d]]:
    #                     actuallogf.write("%f " %(x))
    #                 actuallogf.write("\n")
                # predictionlogf1.close()
                predictionlogf2.close()
                #actuallogf.close()


            # losslogf1.write("%d %f\n" %(i, loss1.item()))
            losslogf2.write("%d %f\n" %(i, loss2.item()))
            # losslogf1.flush()
            losslogf2.flush()
            if np.mod(i, 10) == 0:
                # print("%d Loss_1_%d_%d: %f" %(i,K,m,loss1.item()))
                print("%d Loss_2_%d_%d_%d: %f" %(i,K1, K2,m,loss2.item()))
                #print("Total number of parameters: %d" %(num_params))
            # loss1.backward()
            # optimizer1.step()
            loss2.backward()
            optimizer2.step()
            # no validation process
            # test process
            testloss1 = torch.FloatTensor([0.0])
            testloss2 = torch.FloatTensor([0.0])
            # ys is the final prediction 
            for d in range(0, topdim+1):
                for b in range(0, batch_size):
                    # testloss1 += criterion(ys1[d][b, 0, :], cochain_target_alldegs[d][b, 0, :]) # compare the predicted ones with the underlying true ones only on the sampled ones
                    testloss2 += criterion(ys2[d][b, 0, :], cochain_target_alldegs[d][b, 0, :])
                    # compute the NMSE
            # testlosslogf1.write("%d %f\n" %(i,testloss1))
            # testlosslogf1.flush()
            testlosslogf2.write("%d %f\n" %(i,testloss2))
            testlosslogf2.flush()

        # losslogf1.close()
        losslogf2.close()

        # testlosslogf1.close()
        testlosslogf2.close()

        #name_networks=['C0_1,C0_2','C0_3','C1_1,C1_2','C1_3', 'C2_1,C2_2','C2_3']



if __name__ == "__main__":
    main()
