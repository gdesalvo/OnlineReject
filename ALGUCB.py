import random
import math
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
random.seed(3428)
#random.seed(1234)
############# ############# ############# ############# #############  HELPER FUNCTIONS ############# ############# ############# ############# ############# 

def add_stumps(x1, x2, x3, x4):
    stumps = []
    if x2 - x1 < x3 - x2 and x2 - x1 < x4 - x3:  # R|_|_
        stumps.append([x2, x3, 0])
        stumps.append([x2, x3, 1])
    if x3 - x2 < x2 - x1 and x3 - x2 < x4 - x3:  # _|R|_
        stumps.append([x2, x3, 4])
        stumps.append([x2, x3, 5])
    if x4 - x3 < x2 - x1 and x4 - x3 < x3 - x2:  # _|_|R
        stumps.append([x2, x3, 2])
        stumps.append([x2, x3, 3])
    return stumps

def create_experts(K, want_random,one_d):

    if one_d==0:
         experts=[]
         if want_random:
             for itr in range(K):
                 t1=random.uniform(-1.0,1.0)
                 t2=random.uniform(-1.0,1.0)
                 experts.append([min(t1,t2),max(t1,t2),random.randint(0,5)]) #last number is the stump type
         else:
             t = 2.0 / float(K)
             for itr in np.linspace(-1.0, 1.0, K):
                 if itr > -1.0 and itr < 1.0:
                    if -1.0 + t < itr:
                        experts.extend(add_stumps(-1.0, -1.0 + t, itr, 1.0))
                    if itr + t < 1.0:
                        experts.extend(add_stumps(-1.0, itr, itr + t, 1.0)) 
                    if 1.0 - t > itr:
                        experts.extend(add_stumps(-1.0, itr, 1.0 - t, 1.0))
    elif one_d==1:
        hyp_experts = list(np.linspace(0.0, np.pi, K / 5.0))
        rej_experts = list(np.linspace(0.0, 0.5, 5))
        experts=[]
        for hyp in hyp_experts:
            for rej in rej_experts:
                experts.append([hyp,rej])
    else:
        hyp_experts = list(np.linspace(0.0, 0.5*np.pi, K / 5.0))
        hyp_experts2 = list(np.linspace(0.0, 0.5*np.pi, K / 5.0))
        rej_experts = list(np.linspace(0.0, 0.5, 5))
        experts=[]
        for hyp in hyp_experts:
            for hyp2 in hyp_experts2:
                for rej in rej_experts:
                    experts.append([hyp,hyp2,rej])



    # fexperts=[]
    # rej_experts = [random.uniform(0.0, 1.0) for itr in range(K-1)] # reject if rej_experts[i] < data, 
    # # multiple hypotehsis for each rejecting expert
    # for rej in rej_experts:
    #     for h in range(5):
    #         hyp=random.uniform(0.0, 0.3)
    #         fexperts.append([rej,hyp])
    # hyp_save=[]
    # experts=[]
    # #each hypotehsis has accepting expert
    # for ex in fexperts:
    #     hypo=ex[0]+ex[1]
    #     experts.append([ex[0],ex[1]])
    #     if hypo not in hyp_save:
    #         experts.append([-10.0,hypo]) #always accept for each h
    #         hyp_save.append(hypo)
    # experts.append([10.0,1.0])#always reject one

    return experts

def create_data(T,type_data):
 
    #creating data according to gaussian and labels
    if type_data==0:
        x_data = [random.gauss(0.6, 0.3) for itr in range(T)]
        y_labels = [ int(itr >= 0.5)  for itr in x_data]
#    x_data = [random.uniform(0, 1) for itr in range(T)]
        data = zip(x_data, y_labels)  #data format is list of tuple [(x1,y1),(x2,y2)....]
    elif type_data==1:
        x_data = [[random.uniform(-1.0, 1.0),random.uniform(-1.0,1.0)] for itr in range(T)]
        y_labels = [ int(itr[0] + itr[1] > 0)  for itr in x_data] #label +1 if w_1 x_1 + w_2 x_2 > 0
        data = zip(x_data, y_labels)  #data format is list of tuple [(x1,y1),(x2,y2)....]

    elif type_data==2:
        #load data
        cifar_data = np.genfromtxt('cifar10pca.txt', delimiter=',')
        x_data_comp1=cifar_data[:,0] #first component
        #scale data by max
        x_data=x_data_comp1/float(np.amax(np.absolute(x_data_comp1)))
        cifar_label = np.genfromtxt('cifar10labels.txt', delimiter=',')
        y_labels=[]
        for lab in cifar_label:
            if int(lab)==7:
                y_labels.append(0)
            else:
                y_labels.append(1)
        data = zip(x_data, y_labels)  
        np.random.shuffle(data) #shuffle data
        data=data[:T]

    elif type_data==3:
        new_data = np.genfromtxt('skin.txt', delimiter='\t')
#        x_data_comp1=cifar_data[:,0] #first component
        #scale data by max
        x_data=new_data[:,0:3]
        x_data[:,0]=x_data[:,0]/float(np.amax(np.absolute(x_data[:,0])))
        x_data[:,1]=x_data[:,1]/float(np.amax(np.absolute(x_data[:,1])))
        x_data[:,2]=x_data[:,2]/float(np.amax(np.absolute(x_data[:,2])))

        new_label =new_data[:,-1]


        y_labels=[]
        for lab in new_label:
            if int(lab)==2:
                y_labels.append(0)
            else:
                y_labels.append(1)
        data = zip(x_data.tolist(), y_labels)  
        np.random.shuffle(data) #shuffle data
        data=data[:T]



    return data


def lcb_bound(current_time, pull, alpha):
    if pull == 0:
        return float("inf")
    else:
        return math.sqrt((alpha / 2)* math.log(current_time) / pull)

def loss_of_best_expert(dat,experts,c):
    return min(loss_of_every_expert(dat, experts, c))

def loss_of_every_expert(dat, experts, c,return_rounds,one_d):
    loss_expert_at_rounds=[]
    enum_return_rounds=0
    loss_expert = [0] * len(experts)
    for t in range(len(dat)):
        for i in range(len(experts)):

                loss_expert[i] += rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d), c)
            

        if enum_return_rounds < len(return_rounds) and t+1 ==return_rounds[enum_return_rounds]:
            loss_expert_at_rounds.append([ix / float(t+1) for ix in loss_expert] )
            enum_return_rounds+=1

    return loss_expert_at_rounds


def rej_loss(true_label, expert_label, c):
    if expert_label == -1:
        return c
    else:
        if true_label == expert_label:
            return 0.0
        else:
            return 1.0
                     

def exp_hyp_label(data, expert,one_d):
    if one_d==0:
        if expert[2]==0 or expert[2]==4:
            if expert[1] <= data: 
                expert_label = 1
            else:
                expert_label = 0
        elif expert[2]==1:
            if expert[1] >= data:
                expert_label = 1
            else:
                expert_label = 0
        elif expert[2]==2:
            if expert[0] <= data:
                expert_label = 1
            else:
                expert_label = 0
        elif expert[2]==3  or expert[2]==5:
            if expert[0] >= data:
                expert_label = 1
            else:
                expert_label = 0
        else:
            print 'incorrect stump type'

    elif one_d==1:
        if np.cos(expert[0]) * data[0] + np.sin(expert[0]) * data[1] > 0:
            expert_label = 1
        else:
            expert_label = 0
    else:
        if np.sin(expert[1])*np.cos(expert[0]) * data[0] +np.sin(expert[1])* np.sin(expert[0]) * data[1]- np.cos(expert[1])*data[2] > 0:
            expert_label = 1
        else:
            expert_label = 0

    return expert_label

def exp_label(data, expert,one_d):
    if one_d==0:
        if expert[2]==0 or expert[2]==1: #R|0|1 and R|1|0
            if expert[0] < data:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
               expert_label = -1
        elif expert[2]==2 or expert[2]==3: #0|1|R and 1|0|R 
            if expert[1] > data:
                expert_label = exp_hyp_label(data, expert,one_d)
            else:
               expert_label = -1
        elif expert[2]==4 or expert[2]==5:  #0|R|1 and 1|R|0
            if expert[0]< data < expert[1]:
                expert_label = -1
            else:
                expert_label = exp_hyp_label(data, expert,one_d)

        else:
                print 'incorrect stump type'

    elif one_d==1:
        if  dist_to_plane(data,expert) >= expert[1] : # this is when you accept. if distance to plane is higher than threhsodl

            expert_label = exp_hyp_label(data, expert,one_d)
        else:
           expert_label = -1
    else:
        if  dist_to_plane2(data,expert) >= expert[2] : # this is when you accept. if distance to plane is higher than threhsodl

            expert_label = exp_hyp_label(data, expert,one_d)
        else:
           expert_label = -1

    return expert_label


def dist_to_plane(data,expert):
    # return math.fabs(expert[0]*data[0]-data[1]+0.5)/math.sqrt(expert[0]**2+1+0.5**2)
    return math.fabs(np.cos(expert[0]) * data[0] + np.sin(expert[0]) * data[1])/math.sqrt(np.cos(expert[0])**2+np.sin(expert[0])**2)

def dist_to_plane2(data,expert):
    # return math.fabs(expert[0]*data[0]-data[1]+0.5)/math.sqrt(expert[0]**2+1+0.5**2)
    return math.fabs(np.sin(expert[1])*np.cos(expert[0]) * data[0] + np.sin(expert[1])*np.sin(expert[0]) * data[1]- np.cos(expert[1])*data[2] )/math.sqrt((np.sin(expert[1])*np.cos(expert[0]))**2+(np.sin(expert[1])*np.sin(expert[0]))**2+ np.cos(expert[1])**2)



############# ############# ############# ############# ALGORITHMS ############# ############# ############# ############# ############# ############# 

# def ucb(c, alpha, experts, dat): 

#     K=len(experts)
#     T=len(dat)
#     #initialization step
#     expert_avg=[]
#     loss_alg = 0
#     count_rej=0
#     for i in range(K):
#         expert_label=exp_label(dat[i][0], experts[i])
#         if expert_label==-1:
#             count_rej+=1
#         exp_loss = rej_loss(dat[i][1], expert_label, c)
#         expert_avg.append(exp_loss)
#         loss_alg += exp_loss
#     expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

#     for t in range(K, T):
#         #find best arm
#         lcb_list = [max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient
#         best_arm = lcb_list.index(min(lcb_list)) 

#         expert_pulls[best_arm] += 1 #update number of times arm is pulled

#         #update loss of best arm average
#         inv_pull = 1.0 / expert_pulls[best_arm]
#         expert_loss = rej_loss(dat[t][1], exp_label(dat[t][0], experts[best_arm]), c)
#         expert_avg[best_arm] = expert_loss * inv_pull + (1-inv_pull) * expert_avg[best_arm]
#         if exp_label(dat[t][0],experts[best_arm]) == -1:
#             count_rej+=1

#         #update regret
#         loss_alg += expert_loss

#     return loss_alg / float(T) , count_rej/float(T)


def ucbn(c, alpha, experts, dat, return_rounds, one_d):

    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0
    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0


    for t in range(T):
        #find best armt

        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient
        best_arm = lcb_list.index(min(lcb_list)) 
        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d)
        expert_loss = rej_loss(dat[t][1], expert_label, c) 

#        print best_arm,expert_loss
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'


        if expert_label == -1:
            count_rej+=1
            #update only rejecting experts since "never" receive true label
            for i in range(K): #soinefficient
                if exp_label(dat[t][0], experts[i],one_d) == -1:
                    expert_pulls[i] += 1
                    inv_pull = 1.0 / expert_pulls[i]
                    expert_avg[i] = c * inv_pull + (1 - inv_pull) * expert_avg[i]
        else:
            #update all experts since received true label. 
            for i in range(K): 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                expert_avg[i] = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d), c)  * inv_pull + (1 - inv_pull) * expert_avg[i]
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds





def ucbn_mod(c, alpha, experts, dat, return_rounds, one_d):

    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0
    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0


    for t in range(T):
        #find best arm
        lcb_list=[max(expert_avg[i] , 0.0) for i in range(K)] #soinefficient
        best_arm = lcb_list.index(min(lcb_list)) 
        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d)
        expert_loss = rej_loss(dat[t][1], expert_label, c) 

#        print best_arm,expert_loss
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'
        if expert_label == -1:
            count_rej+=1
            #update only rejecting experts since "never" receive true label
            for i in range(K): #soinefficient
                if exp_label(dat[t][0], experts[i],one_d) == -1:
                    expert_pulls[i] += 1
                    inv_pull = 1.0 / expert_pulls[i]
                    expert_avg[i] = c * inv_pull + (1 - inv_pull) * expert_avg[i]
        else:
            #update all experts since received true label. 
            for i in range(K): 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                current_loss = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d), c) 
                expert_avg[i] = current_loss * inv_pull + (1 - inv_pull) * expert_avg[i]
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbmax(c, alpha, experts, dat, return_rounds, one_d):

    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0
    K = len(experts)
    T = len(dat)
#    print '\n\n UCB N'
    expert_avg = [0.0]*K
    expert_pulls = [0.0] * K 
    loss_alg = 0
    count_rej=0


    for t in range(T):
        #find best arm
        lcb_list=[max(expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha), 0.0) for i in range(K)] #soinefficient
        best_lcb_arm = lcb_list.index(min(lcb_list)) 
        #update regret
        best_lcb_expert_label = exp_label(dat[t][0], experts[best_lcb_arm],one_d)
        if best_lcb_expert_label==-1:
            #if rejecting, then one with smallest expert_avg b/c will always see best_lcb_expert gettin updated
            best_arm=expert_avg.index(min(expert_avg))
        else:
            #if bestlcb is accepting then pick out of accepting ones the one with smallest emp mean
            best_arm=best_lcb_expert_label
            for i in range(K):
                if exp_label(dat[t][0],experts[i],one_d)!=-1 and expert_avg[i] < expert_avg[best_arm]: 
                    best_arm=i
                
#        best_lcb_expert_loss = rej_loss(dat[t][1], expert_label, c) 
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d)
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
#        print best_arm,expert_loss
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'
        if expert_label == -1:
            count_rej+=1
            #update only rejecting experts since "never" receive true label
            for i in range(K): #soinefficient
                if exp_label(dat[t][0], experts[i],one_d) == -1:
                    expert_pulls[i] += 1
                    inv_pull = 1.0 / expert_pulls[i]
                    expert_avg[i] = c * inv_pull + (1 - inv_pull) * expert_avg[i]
        else:
            #update all experts since received true label. 
            for i in range(K): 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                current_loss = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i],one_d), c) 
                expert_avg[i] = current_loss * inv_pull + (1 - inv_pull) * expert_avg[i]
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbth(c, alpha, experts, dat,return_rounds,one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = [0.0]*K #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0
    num_acc=0

    
    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t

        acc_avg_set = []
        best_acc_exp=-1
        best_acc_avg=2.0
        rej_exp = 1.0
        best_rej_exp=-1

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                acc_avg_set.append(hyp_expert_avg[i])
                if hyp_expert_avg[i]< best_acc_avg:
                    best_acc_avg = hyp_expert_avg[i]
                    best_acc_exp = i

#                if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < acc_exp:
#                    acc_exp= max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
#                    best_acc_exp=i
            else:
                if hyp_expert_avg[i] < rej_exp:
                    rej_exp=hyp_expert_avg[i] 
                    best_rej_exp=i
                    

        #find best arm
        if  best_acc_exp!=-1 and max(np.mean(np.array(acc_avg_set))- lcb_bound(t,num_acc,alpha),0)<=c:
                best_arm = best_acc_exp
                num_acc+=1
        else:
            if best_rej_exp!=-1:
                best_arm = best_rej_exp
                count_rej+=1
            else:
                best_arm = best_acc_exp
                num_acc+=1
        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        #print best_arm, expert_loss,expert_pulls[best_arm]
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds

def ucbtth(c, alpha, experts, dat,return_rounds,one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = [0.0]*K #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0
    num_acc=0

    
    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t

        acc_avg_set = []
        best_acc_exp=-1
        best_acc_avg=10.0
        rej_exp = 1.0
        best_rej_exp=-1

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                acc_avg_set.append(hyp_expert_avg[i])
                if max(hyp_expert_avg[i]- lcb_bound(t,expert_pulls[i],alpha), 0.0)< best_acc_avg:
                    best_acc_avg = max(hyp_expert_avg[i]- lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_acc_exp = i

#                if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < acc_exp:
#                    acc_exp= max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
#                    best_acc_exp=i
            else:
                if max(hyp_expert_avg[i]- lcb_bound(t,expert_pulls[i],alpha), 0.0) < rej_exp:
                    rej_exp=max(hyp_expert_avg[i]- lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_rej_exp=i
                    

        #find best arm
        if  best_acc_exp!=-1 and max(np.mean(np.array(acc_avg_set))- lcb_bound(t,num_acc,alpha),0)<=c:
                best_arm = best_acc_exp
                num_acc+=1
        else:
            if best_rej_exp!=-1:
                best_arm = best_rej_exp
                count_rej+=1
            else:
                best_arm = best_acc_exp
                num_acc+=1
        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        #print best_arm, expert_loss,expert_pulls[best_arm]
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds





def ucbcc(c, alpha, experts, dat,return_rounds, one_d):

    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)

    expert_cnt_acc = [0.0] * K
    expert_pulls = [0.0] * K
    expert_hyp_losses = [0.0] * K
    loss_alg = 0
    count_rej=0


    for t in range(T):

        # aggregate lcbs
        expert_lcbs = []
        # update slacks and empirical averages
        for i in range(K):
            if exp_label(dat[t][0], experts[i],one_d) != -1:  # expert i accepts
                expert_cnt_acc[i] += 1

        for i in range(K):
            exp_prob_acc = expert_cnt_acc[i] / (t+1)
            exp_prob_rej = 1 - exp_prob_acc
            if exp_label(dat[t][0],experts[i],one_d)!=-1:
                expert_lcbs.append(max(exp_prob_acc ,0.0) * max((expert_hyp_losses[i] / max(expert_pulls[i],1)),0.0))
            else:
                expert_lcbs.append(max(exp_prob_rej , 0.0) * c) 
        
        #find best arm
        best_arm = expert_lcbs.index(min(expert_lcbs)) 
        
        #update algorithm loss
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d)
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        loss_alg += expert_loss

        if expert_label != -1:  # so that we see the label
            for i in range(K):
                expert_pulls[i] += 1
                current_label = exp_hyp_label(dat[t][0], experts[i],one_d)
                current_loss = rej_loss(dat[t][1], current_label, c)
                expert_hyp_losses[i] += current_loss 
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbd(c, alpha, experts, dat,return_rounds, one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)

    expert_cnt_acc = [0.0] * K
    expert_pulls = [0.0] * K
    expert_hyp_losses = [0.0] * K
    loss_alg = 0
    count_rej=0


    for t in range( T):

        # aggregate lcbs
        expert_lcbs = []
        # update slacks and empirical averages
        for i in range(K):
            if exp_label(dat[t][0], experts[i],one_d) != -1:  # expert i accepts
                expert_cnt_acc[i] += 1

        for i in range(K):
            exp_prob_acc = expert_cnt_acc[i] / (t+1)
            exp_prob_rej = 1 - exp_prob_acc
            expert_lcbs.append(max(exp_prob_acc - lcb_bound(t+1, t+1, alpha), 0.0) * max((expert_hyp_losses[i] / max(expert_pulls[i],1)) - lcb_bound(t+1, expert_pulls[i], alpha), 0.0) + max(exp_prob_rej - lcb_bound(t+1, t+1, alpha), 0.0) * c) 
        
        #find best arm
        best_arm = expert_lcbs.index(min(expert_lcbs)) 
        
        #update algorithm loss
        expert_label = exp_label(dat[t][0], experts[best_arm],one_d)
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        loss_alg += expert_loss

        if expert_label != -1:  # so that we see the label
            for i in range(K):
                expert_pulls[i] += 1
                current_label = exp_hyp_label(dat[t][0], experts[i],one_d)
                current_loss = rej_loss(dat[t][1], current_label, c)
                expert_hyp_losses[i] += current_loss 
        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds
       
def ucbh(c, alpha, experts, dat,return_rounds,one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = [0.0]*K #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0

    
    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t

        acc_exp = 1.0
        best_acc_exp=-1
        rej_exp = 1.0
        best_rej_exp=-1

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < acc_exp:
                    acc_exp= max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_acc_exp=i
            else:
                if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < rej_exp:
                    rej_exp=max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_rej_exp=i
                    

        #find best arm
        if acc_exp < c and best_acc_exp!=-1:
                best_arm = best_acc_exp
        else:
            if best_rej_exp!=-1:
                best_arm = best_rej_exp
                count_rej+=1
            else:
                best_arm = best_acc_exp

        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        #print best_arm, expert_loss,expert_pulls[best_arm]
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds

def ucbhnew_mod(c, alpha, experts, dat,return_rounds,one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = [0.0]*K #only care of emperical average over hyp_experts.
    est_loss=[0.0]*K
    seen_loss=[0.0]*K
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0

    
    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t

        acc_exp = 1.0
        best_acc_exp=-1
        rej_exp = 1.0
        best_rej_exp=-1

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < acc_exp and hyp_expert_avg[i]  <= min(est_loss) :
                    acc_exp= max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_acc_exp=i
            else:
                if max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < rej_exp:
                    rej_exp=max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_rej_exp=i
                    

        #find best arm
        if acc_exp < c and best_acc_exp!=-1:
                best_arm = best_acc_exp
        else:
            if best_rej_exp!=-1:
                best_arm = best_rej_exp
                count_rej+=1
            else:
                best_arm = best_acc_exp

        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        #print best_arm, expert_loss,expert_pulls[best_arm]
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]
        
        for kk in range(K):
            if best_expert_label!= -1:
                seen_loss[kk]+=1
                inv_seen= 1.0/seen_loss[kk]
                est_loss[kk] = rej_loss(dat[t][1], save_expert_labels[kk], c)*inv_seen + (1.0-inv_seen) * est_loss[kk] 
            else:
                if save_expert_labels[kk]==-1:
                    seen_loss[kk]+=1
                    inv_seen= 1.0/seen_loss[kk]
                    est_loss[kk]+=c*inv_seen+ (1.0-inv_seen)*est_loss[kk]

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbhnew(c, alpha, experts, dat,return_rounds,one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = [[0.0]*K]*K #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0

    
    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t

        acc_exp = 1.0
        best_acc_exp=-1
        old_acc_exp = 1.0
        old_best_acc_exp=-1

        rej_exp = 1.0
        best_rej_exp=-1

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if max(hyp_expert_avg[i][i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < acc_exp and ( min(hyp_expert_avg[best_acc_exp])== hyp_expert_avg[best_acc_exp][best_acc_exp] or expert_pulls[i] < 10):
                        acc_exp= max(hyp_expert_avg[i][i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                        best_acc_exp=i
                if max(hyp_expert_avg[i][i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < old_acc_exp: 
                        old_acc_exp= max(hyp_expert_avg[i][i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                        old_best_acc_exp=i

            else:
                if max(hyp_expert_avg[i][i] - lcb_bound(t,expert_pulls[i],alpha), 0.0) < rej_exp:
                    rej_exp=max(hyp_expert_avg[i][i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    best_rej_exp=i

        #find best arm
        if acc_exp < c and best_acc_exp!=-1:
                best_arm = best_acc_exp
        else:
            if best_rej_exp!=-1:
                best_arm = best_rej_exp
                count_rej+=1
            else:
                best_arm= old_best_acc_exp

        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        #print best_arm, expert_loss,expert_pulls[best_arm]
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            for ii in range(K):
                                hyp_expert_avg[jj][ii] = rej_loss(dat[t][1], exp_label(dat[t][0], experts[ii],one_d) , c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj][ii]

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds




def ucbh_mod(c, alpha, experts, dat,return_rounds,one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = [0.0]*K #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0

    
    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t

        acc_exp = 1.0
        best_acc_exp=-1
        rej_exp = 1.0
        best_rej_exp=-1

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if max(hyp_expert_avg[i], 0.0) < acc_exp:
                    acc_exp= max(hyp_expert_avg[i] , 0.0)
                    best_acc_exp=i
            else:
                if max(hyp_expert_avg[i] , 0.0) < rej_exp:
                    rej_exp=max(hyp_expert_avg[i] , 0.0)
                    best_rej_exp=i
                    

        #find best arm
        if acc_exp < c and best_acc_exp!=-1:
                best_arm = best_acc_exp
        else:
            if best_rej_exp!=-1:
                best_arm = best_rej_exp
                count_rej+=1
            else:
                best_arm = best_acc_exp

        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        #print best_arm, expert_loss,expert_pulls[best_arm]
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds




# def ucbh_mod(c, alpha, experts, dat,return_rounds,one_d):

#     loss_alg_at_return_rounds=[]
#     count_rej_at_return_rounds=[]
#     enum_return_rounds=0

#     K = len(experts)
#     T = len(dat)
# #    print '\n\n ucbH'
#     hyp_expert_avg = {} #only care of emperical average over hyp_experts.
#     expert_pulls = [0.0]*K #counts the number of times when r>0
#     loss_alg = 0
#     count_rej=0
#     #initialization step
#     # for i in range(K):
#     #     expert_label = exp_label(dat[i][0], experts[i])
#     #     if expert_label != -1:
#     #         exp_loss = rej_loss(dat[i][1], expert_label, c) 
#     #         hyp_expert_avg[str(i)] = exp_loss  #prob should define zero/one loss and use it here
#     #         loss_alg += exp_loss
#     #         expert_pulls.append(1)
#     #     else:
#     #         expert_pulls.append(0)
#     #         loss_alg += c
#     #         count_rej+=1
    
#     for t in range(T):
#         #use dictionary so keep track of which are accepting and rejecting experts at time t
#         acc_exp = {}  
#         rej_exp = {}

#         save_expert_labels=[]
#         for i in range(K):
#             #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
#             expert_label = exp_label(dat[t][0], experts[i],one_d)
#             save_expert_labels.append(expert_label)
#             if expert_label!=-1:
#                 if str(i) in hyp_expert_avg.keys():
#                     acc_exp[str(i)] = max(hyp_expert_avg[str(i)], 0.0)
#                 else:
#                     acc_exp[str(i)] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf
#             else:
#                 if str(i) in hyp_expert_avg.keys():
#                     rej_exp[str(i)] = max(hyp_expert_avg[str(i)], 0.0)
#                 else:
#                     rej_exp[str(i)] = 0.0  # -float("inf")
                    

#         #find best arm
#         if bool(acc_exp) == True and acc_exp[min(acc_exp, key=acc_exp.get)] < c:
#                 best_arm = int(min(acc_exp, key=acc_exp.get))
#         else:
#             if bool(rej_exp) == True:
#                 best_arm = int(min(rej_exp, key=rej_exp.get))
#                 count_rej+=1
#             else:
#                 best_arm = int(min(acc_exp, key=acc_exp.get))

        
#         #update regret
#         best_expert_label = save_expert_labels[best_arm]
#         expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
#         #print best_arm, expert_loss,expert_pulls[best_arm]
#         loss_alg += expert_loss
# #        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
#         if best_expert_label != -1:
#             # if pulled a nonrej expert update all acc_experts
#             for jj in range(len(save_expert_labels)):
#                 if save_expert_labels[jj] != -1:
#                         if str(jj) in hyp_expert_avg.keys():
#                             expert_pulls[jj] += 1
#                             inv_pull = 1.0 / expert_pulls[jj]
#                             hyp_expert_avg[str(jj)] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[str(jj)]
#                         else:
#                             expert_pulls[jj] += 1
#                             hyp_expert_avg[str(jj)] = rej_loss(dat[t][1], save_expert_labels[jj], c)
#         if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
#             loss_alg_at_return_rounds.append(loss_alg/float(t+1))
#             count_rej_at_return_rounds.append(count_rej/float(t+1))
#             enum_return_rounds+=1
#     return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbt(c, alpha, experts, dat,return_rounds, one_d):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    gamma=0.9
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = {} #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0

    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t
        acc_exp = {}  
        good_acc_exp={}
        rej_exp = {}
        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i],one_d)
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if 1.0-(expert_pulls[i])/(t+1) <= gamma:  # fractin of times r rejected is less than gamma
                    if str(i) in hyp_expert_avg.keys():
                        good_acc_exp[str(i)] = max(hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    else:
                        #i dont think this every happens
                        good_acc_exp[str(i)] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf

                if str(i) in hyp_expert_avg.keys():
                        acc_exp[str(i)] = max(hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                else:
                        acc_exp[str(i)] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf
            else:
                if str(i) in hyp_expert_avg.keys():
                    rej_exp[str(i)] = max(hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                else:
                    rej_exp[str(i)] = 0.0  # -float("inf")
                    

        #find best arm
        if bool(good_acc_exp) == True and good_acc_exp[min(good_acc_exp, key=good_acc_exp.get)] < c:
                best_arm = int(min(good_acc_exp, key=good_acc_exp.get))
                
        else:
            if bool(rej_exp) == True:
                best_arm = int(min(rej_exp, key=rej_exp.get))
                count_rej+=1
            else:
                #if no good accepting experts and no rejecting experts pick some random accepting expert
                best_arm = int(min(acc_exp, key=acc_exp.get))

        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                        if str(jj) in hyp_expert_avg.keys():
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[str(jj)] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[str(jj)]
                        else:
                            expert_pulls[jj] += 1
                            hyp_expert_avg[str(jj)] = rej_loss(dat[t][1], save_expert_labels[jj], c)

        if enum_return_rounds < len(return_rounds) and t+1==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds



def ucbvt(c, alpha, experts, dat,return_rounds, one_d):

    # YOU CAN'T PARALLELIZE THIS IN T BECAUSE YOU T IN ALGORITHM
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    for T in return_rounds:
        K = len(experts)
#        T = len(dat)
        hyp_expert_avg = {} #only care of emperical average over hyp_experts.
        expert_pulls = [0.0]*K #counts the number of times when r>0
        loss_alg = 0
        count_rej=0

        for t in range(T):
            #use dictionary so keep track of which are accepting and rejecting experts at time t
            acc_exp = {}  
            good_acc_exp={}
            rej_exp = {}
            save_expert_labels=[]
            for i in range(K):
                #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
                expert_label = exp_label(dat[t][0], experts[i],one_d)
                save_expert_labels.append(expert_label)
                if expert_label!=-1:
                    if 1.0-(expert_pulls[i])/(t+1) <= ( 1.0-1.0/float(T) ):  # fractin of times r rejected is less than gamma
                        if str(i) in hyp_expert_avg.keys():
                            good_acc_exp[str(i)] = max(hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                        else:
                            #i dont think this every happens
                            good_acc_exp[str(i)] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf

                    if str(i) in hyp_expert_avg.keys():
                            acc_exp[str(i)] = max(hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    else:
                            acc_exp[str(i)] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf
                else:
                    if str(i) in hyp_expert_avg.keys():
                        rej_exp[str(i)] = max(hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    else:
                        rej_exp[str(i)] = 0.0  # -float("inf")


            #find best arm
            if bool(good_acc_exp) == True and good_acc_exp[min(good_acc_exp, key=good_acc_exp.get)] < c:
                    best_arm = int(min(good_acc_exp, key=good_acc_exp.get))

            else:
                if bool(rej_exp) == True:
                    best_arm = int(min(rej_exp, key=rej_exp.get))
                    count_rej+=1
                else:
                    best_arm = int(min(acc_exp, key=acc_exp.get))


            #update regret
            best_expert_label = save_expert_labels[best_arm]
            expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
            #print best_arm, expert_loss,expert_pulls[best_arm]
            loss_alg += expert_loss
    #        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
            if best_expert_label != -1:
                # if pulled a nonrej expert update all acc_experts
                for jj in range(len(save_expert_labels)):
                    if save_expert_labels[jj] != -1:
                            if str(jj) in hyp_expert_avg.keys():
                                expert_pulls[jj] += 1
                                inv_pull = 1.0 / expert_pulls[jj]
                                hyp_expert_avg[str(jj)] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[str(jj)]
                            else:
                                expert_pulls[jj] += 1
                                hyp_expert_avg[str(jj)] = rej_loss(dat[t][1], save_expert_labels[jj], c)



        loss_alg_at_return_rounds.append(loss_alg / float(T))    
        count_rej_at_return_rounds.append(count_rej / float(T))    

    return loss_alg_at_return_rounds,count_rej_at_return_rounds




############# ############# ############# ############# #############  PLOTTING ############# ############# ############# ############# ############# 
def plotting(c,alpha,K,text_file,ONE_D,TYPE_DATA):
#NEED TO IMRPOVE THIS PLOTTING FUNCTN 
    NUM_AVG=2
    T_MAX=700
    avg_regret=[]
    avg_counts=[]
    avg_losses=[]
    for er in range(NUM_AVG):
            experts= create_experts(K, False, ONE_D)
            print len(experts)
            x=range(200,T_MAX,200) 

            loss=[]
            count=[]
            expert_loss=[]

            for p in range(1):
                data=create_data(T_MAX,TYPE_DATA)
                loss_experts=loss_of_every_expert(data,experts,c,x,ONE_D)                
                loss1,countrej1=ucbn(c,alpha,experts,data,x,ONE_D) #returns values of all needed roudns
                loss2,countrej2=ucbd(c,alpha,experts,data,x,ONE_D)
                loss3,countrej3=ucbh(c,alpha,experts,data,x,ONE_D)
#                loss4,countrej4=ucbhnew_mod(c,alpha,experts,data,x,ONE_D)
#                loss5,countrej5=ucbh_mod(c,alpha,experts,data,x,ONE_D)
#                loss6,countrej6=ucbd(c,alpha,experts,data,x,ONE_D)
#                loss7,countrej7=ucbtth(c,alpha,experts,data,x,ONE_D)
                expert_loss.append(loss_experts)
                loss.append([loss1,loss2,loss3])#,loss4])#,loss5,loss6,loss7])
                count.append([countrej1,countrej2,countrej3])#,countrej4])#,countrej5,countrej6,countrej7])
            

            loss=np.mean(np.array(loss),axis=0)
            count=np.mean(np.array(count),axis=0)
            expert_loss=np.mean(np.array(expert_loss),axis=0)


            best_expert_loss=np.amin(expert_loss,axis=1)
            regret=loss-np.expand_dims(best_expert_loss,axis=0)
            avg_regret.append(regret)
            avg_losses.append(loss)
            avg_counts.append(count)

    std_regret=np.std(np.array(avg_regret),axis=0)
    avg_regret=np.mean(np.array(avg_regret),axis=0)
    std_losses=np.std(np.array(avg_losses),axis=0)
    avg_losses=np.mean(np.array(avg_losses),axis=0)  
    std_counts=np.std(np.array(avg_counts),axis=0)
    avg_counts=np.mean(np.array(avg_counts),axis=0)



    text_file.write('\nPseudo Regret of UCB-type Algorithms for '+str(K)+' arms with c '+str(c)+'_dimension_'+str(int(ONE_D)))
    text_file.write('; regret UCBN:'+str(avg_regret[0])+'; std UCBN:'+str(std_regret[0]))
    text_file.write('; regret UCBD:'+str(avg_regret[1])+'; std UCBD:'+str(std_regret[1]))
    text_file.write('; regret UCBH:'+str(avg_regret[2])+'; std UCBH:'+str(std_regret[2]))
#    text_file.write('; regret UCBHNEW:'+str(avg_regret[3])+'; std UCBHNEW:'+str(std_regret[3]))
#    text_file.write('; regret UCBH_MOD:'+str(avg_regret[4])+'; std UCBH_MOD:'+str(std_regret[4]))
#    text_file.write('; regret UCBD:'+str(avg_regret[5])+'; std UCBD:'+str(std_regret[5]))
#    text_file.write('; regret UCBTTH:'+str(avg_regret[6])+'; std UCBTTH:'+str(std_regret[6]))

    text_file.write('; losses UCBN:'+str(avg_losses[0])+'; std UCBN:'+str(std_losses[0]))
    text_file.write('; losses UCBD:'+str(avg_losses[1])+'; std UCBD:'+str(std_losses[1]))
    text_file.write('; losses UCBH:'+str(avg_losses[2])+'; std UCBH:'+str(std_losses[2]))#
#    text_file.write('; losses UCBHNEW:'+str(avg_losses[3])+'; std UCBHNEW:'+str(std_losses[3]))
#    text_file.write('; losses UCBH_MOD:'+str(avg_losses[4])+'; std UCBH_MOD:'+str(std_losses[4]))
#    text_file.write('; losses UCBD:'+str(avg_losses[5])+'; std UCBD:'+str(std_losses[5]))
#    text_file.write('; losses UCBTTH:'+str(avg_losses[6])+'; std UCBTTH:'+str(std_losses[6]))

    text_file.write('; counts UCBN:'+str(avg_counts[0])+'; std UCBN:'+str(std_counts[0]))
    text_file.write('; counts UCBD:'+str(avg_counts[1])+'; std UCBD:'+str(std_counts[1]))
    text_file.write('; counts UCBH:'+str(avg_counts[2])+'; std UCBH:'+str(std_counts[2]))
#    text_file.write('; counts UCBHNEW:'+str(avg_counts[3])+'; std UCBHNEW:'+str(std_counts[3]))
#    text_file.write('; counts UCBH_MOD:'+str(avg_counts[4])+'; std UCBH_MOD:'+str(std_counts[4]))
#    text_file.write('; counts UCBD:'+str(avg_counts[5])+'; std UCBD:'+str(std_counts[5]))
#    text_file.write('; counts UCBTTH:'+str(avg_counts[6])+'; std UCBTTH:'+str(std_counts[6]))


    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.set_xscale("log", nonposx='clip')
#    ax.set_yscale("log", nonposy='clip')
    ax.errorbar(x, avg_regret[0], yerr=std_regret[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_regret[1], yerr=std_regret[1],fmt='-',color='limegreen', label='UCB-D')
    ax.errorbar(x, avg_regret[2], yerr=std_regret[2],fmt='b-', label='UCB-H')
#    ax.errorbar(x, avg_regret[3], yerr=std_regret[3],fmt='r-', label='UCB-HNEW_MOD')
#    ax.errorbar(x, avg_regret[4], yerr=std_regret[4],fmt='c-', label='UCB-H_MOD')
#    ax.errorbar(x, avg_regret[5], yerr=std_regret[5],fmt='y-', label='UCB-D')
#    ax.errorbar(x, avg_regret[6], yerr=std_regret[6],fmt='m-', label='UCB-TTH')
#    ax.axhline(y=0.0,c="purple",linewidth=2,zorder=10)
#    legend = ax.legend(loc='upper right', shadow=True)
#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#    ax.legend(loc='center left', bbox_to_anchor=(1.5, 1))
    plt.xlabel('Rounds')
    plt.ylabel('Pseudo-Regret')
    plt.title('Pseudo-Regret for c = '+str(c),y=1.02)
    plt.savefig('./regret_K'+str(len(experts))+'_c'+str(c)+'_dimension_'+str(int(ONE_D))+'.eps', format='eps', dpi=1000,bbox_inches='tight')
#    plt.savefig('./regret_K'+str(len(experts))+'_c'+str(c)+'_dimension_'+str(int(ONE_D))+'.png',bbox_inches='tight')

    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.set_xscale("log", nonposx='clip')
 #   ax.set_yscale("log", nonposy='clip')
    ax.errorbar(x, avg_losses[0], yerr=std_losses[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_losses[1], yerr=std_losses[1],fmt='-',color='limegreen', label='UCB-D')
    ax.errorbar(x, avg_losses[2], yerr=std_losses[2],fmt='b-', label='UCB-H')
#    ax.errorbar(x, avg_losses[3], yerr=std_losses[3],fmt='r-', label='UCB-HNEW_MOD')
#    ax.errorbar(x, avg_losses[4], yerr=std_losses[4],fmt='c-', label='UCB-H_MOD')
#    ax.errorbar(x, avg_losses[5], yerr=std_losses[5],fmt='y-', label='UCB-D')
#    ax.errorbar(x, avg_losses[6], yerr=std_losses[6],fmt='m-', label='UCB-TTH')
#    ax.axhline(y=0.0,c="purple",linewidth=2,zorder=10)
#    legend = ax.legend(loc='upper right', shadow=True)
#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#    ax.legend(loc='center left', bbox_to_anchor=(1.5, 1))
    plt.xlabel('Rounds')
    plt.ylabel(' Losses')
    plt.title('Losses for c: '+str(c),y=1.02)
    plt.savefig('./losses_K'+str(len(experts))+'_c'+str(c)+'_dimension_'+str(int(ONE_D))+'.eps', format='eps', dpi=1000,bbox_inches='tight')
#    plt.savefig('./losses_K'+str(len(experts))+'_c'+str(c)+'_dimension_'+str(int(ONE_D))+'.png',bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xscale("log", nonposx='clip')
  #  ax.set_yscale("log", nonposy='clip')
    ax.errorbar(x, avg_counts[0], yerr=std_counts[0],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_counts[1], yerr=std_counts[1],fmt='-',color='limegreen', label='UCB-D')
    ax.errorbar(x, avg_counts[2], yerr=std_counts[2],fmt='b-', label='UCB-H')
#    ax.errorbar(x, avg_counts[3], yerr=std_counts[3],fmt='r-', label='UCB-HNEW_MOD')
#    ax.errorbar(x, avg_counts[4], yerr=std_counts[4],fmt='c-', label='UCB-H_MOD')
#    ax.errorbar(x, avg_counts[5], yerr=std_counts[5],fmt='y-', label='UCB-D')
#    ax.errorbar(x, avg_counts[6], yerr=std_counts[6],fmt='m-', label='UCB-TTH')
 #   legend = ax.legend(loc='upper right', shadow=True)
#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#    ax.legend(loc='center left', bbox_to_anchor=(1.5, 1))

    plt.xlabel('Rounds')
    plt.ylabel('Fraction of Rejection')
    plt.title('Fraction of Rejection for '+str(len(experts))+' arms with c '+str(c))
    plt.savefig('./counts_K'+str(len(experts))+'_c'+str(c)+'_dimension_'+str(int(ONE_D))+'.png', dpi=1000,bbox_inches='tight')


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"
############# ############# ############# ############# #############  MAIN ############# ############# ############# ############# ############# 
if __name__ == "__main__":
    tic()
    alpha=3
    val=int(sys.argv[1])
#    K_values=[602,302,202,102,52] #for 2d
    K_values=[102,52,77,27,17] # for one d
#    c_values=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    c_values=[0.1,0.2,0.3,0.1,0.2,0.3,0.1,0.2,0.3]
    
    if val <=3:
        K=19
        c=c_values[val-1] 
        ONE_D=0   #one_d determines if you want to use 1d experts (0) vs 2d experts (1) vs 3d experts(2)
        TYPE_DATA=2  #type_data determines if you want (0) 1d data drawn from gaussian(0.6,0.3), (1) 2d data drawn uniformly on [-1,1]X[-1,1] square, (2) loads 1d cifar data set

    elif 3<val<=6:
        K=602
        c=c_values[val-1] 
        ONE_D=1  #one_d determines if you want to use 1d experts (0) vs 2d experts (1) vs 3d experts(2)
        TYPE_DATA=1  #type_data determines if you want (0) 1d data drawn from gaussian(0.6,0.3), (1) 2d data drawn uniformly on [-1,1]X[-1,1] square, (2) loads 1d cifar data set
    else:
        K=55
        c=c_values[val-1] 
        ONE_D=2
        TYPE_DATA=3
#    print K
#    print c
#    print ONE_D
#    print TYPE_DATA
#    if val<=18:
#        K=K_values[0]
#        c= c_values[val-1]
#    elif 18<val and val<=36:
#        K=K_values[1]
#        c= c_values[val-19]
#    elif 36 <val and val <=54:
#        K=K_values[2]
#        c= c_values[val-37]
#    elif 54 <val and val <=72:
#        K=K_values[3]
#        c= c_values[val-55]
#    elif 72 <val and val <=90:
#        K=K_values[4]
#        c= c_values[val-73]
#    else:
#        print 'too high values'

    text_file = open("./Output_" + str(K) + "arms.txt", "w")
    plotting(c,alpha,K,text_file,ONE_D,TYPE_DATA) #last plot point is for T=2000                   
    text_file.close()


    toc()
