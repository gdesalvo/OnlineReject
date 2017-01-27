import random
import math
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
random.seed(3428)
#random.seed(1234)
############# ############# ############# ############# #############  HELPER FUNCTIONS ############# ############# ############# ############# ############# 
def create_experts(K, want_random):
    #creating the rejecting and hypothesis experts according to threholds
#    if want_random:
#        rej_experts = [random.uniform(0.0, 1.0) for itr in range(K)] # reject if rej_experts[i] < data, 
#    else:
#        rej_experts = list(np.linspace(0.0, 1.0, K))


#    if want_random:
#        hyp_experts = [random.uniform(0.0, 0.3) for itr in range(K)]  # classification surface is given by threhold rej_expert[i]+hyp_expert[i]
#    else:
#        hyp_experts = list(np.linspace(0.0, 0.3, K))
#    experts = zip(rej_experts, hyp_experts) #expert format is list of tuple [(rej_threshold1, hyp_threshold1),(rej_threshold2, hyp_threshold2).... ]    

    hyp_experts=list(np.linspace(-2.0, 0.0, K/5.0))  #experts are of the form w*x_1+0.05-x_2=0 so we just specify w
    rej_experts=list(np.linspace(0.0, 0.5, 5))  #5 confidence based thresholds for each w
    experts=[]
    for hyp in hyp_experts:
        for rej in rej_experts:
            experts.append([hyp,rej])


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

def create_data(T):
 
    #creating data according to gaussian and labels
#    x_data = [random.gauss(0.6, 0.1) for itr in range(T)]
#    y_labels = [ int(itr >= 0.6)  for itr in x_data]
    x_data = [[random.uniform(0.0, 1.0),random.uniform(0.0,1.0)] for itr in range(T)]
    y_labels = [ int(-itr[0]+0.5 <= itr[1])  for itr in x_data] #label +1 above -x+0.5

    data = zip(x_data, y_labels)  #data format is list of tuple [(x1,y1),(x2,y2)....]
    return data


def lcb_bound(current_time, pull, alpha):
    if pull == 0:
        return float("inf")
    else:
        return math.sqrt((alpha / 2)* math.log(current_time) / pull)

def loss_of_best_expert(dat,experts,c):
    return min(loss_of_every_expert(dat, experts, c))

def loss_of_every_expert(dat, experts, c,return_rounds):
    loss_expert_at_rounds=[]
    enum_return_rounds=0
    loss_expert = [0] * len(experts)
    for t in range(len(dat)):
        for i in range(len(experts)):
                loss_expert[i] += rej_loss(dat[t][1], exp_label(dat[t][0], experts[i]), c)
            

        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
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
                     

def exp_hyp_label(data, expert):
    if expert[0]*data[0] +0.5 - data[1] <0:
        expert_label = 1
    else:
        expert_label = 0
    return expert_label

def exp_label(data, expert):
    if  dist_to_plane(data,expert) >= expert[1] : # this is when you accept. if distance to plane is higher than threhsodl
        expert_label = exp_hyp_label(data, expert)
    else:
       expert_label = -1
    return expert_label

def dist_to_plane(data,expert):
    return math.fabs(expert[0]*data[0]-data[1]+0.5)/math.sqrt(expert[0]**2+1+0.5**2)


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


def ucbn(c, alpha, experts, dat, return_rounds):

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
        best_arm = lcb_list.index(min(lcb_list)) 
        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm])
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbN'
        if expert_label == -1:
            count_rej+=1
            #update only rejecting experts since "never" receive true label
            for i in range(K): #soinefficient
                if exp_label(dat[t][0], experts[i]) == -1:
                    expert_pulls[i] += 1
                    inv_pull = 1.0 / expert_pulls[i]
                    expert_avg[i] = c * inv_pull + (1 - inv_pull) * expert_avg[i]
        else:
            #update all experts since received true label. 
            for i in range(K): 
                expert_pulls[i] += 1
                inv_pull = 1.0 / expert_pulls[i]
                current_loss = rej_loss(dat[t][1], exp_label(dat[t][0], experts[i]), c) 
                expert_avg[i] = current_loss * inv_pull + (1 - inv_pull) * expert_avg[i]
        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbcc(c, alpha, experts, dat,return_rounds):

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
            if exp_label(dat[t][0], experts[i]) != -1:  # expert i accepts
                expert_cnt_acc[i] += 1

        for i in range(K):
            exp_prob_acc = expert_cnt_acc[i] / (t+1)
            exp_prob_rej = 1 - exp_prob_acc
            if exp_label(dat[t][0],experts[i])!=-1:
                expert_lcbs.append(max(exp_prob_acc - lcb_bound(t+1, t+1, alpha),0.0) * max((expert_hyp_losses[i] / max(expert_pulls[i],1)) - lcb_bound(t+1, expert_pulls[i], alpha),0.0))
            else:
                expert_lcbs.append(max(exp_prob_rej - lcb_bound(t+1, t+1, alpha), 0.0) * c) 
        
        #find best arm
        best_arm = expert_lcbs.index(min(expert_lcbs)) 
        
        #update algorithm loss
        expert_label = exp_label(dat[t][0], experts[best_arm])
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        loss_alg += expert_loss

        if expert_label != -1:  # so that we see the label
            for i in range(K):
                expert_pulls[i] += 1
                current_label = exp_hyp_label(dat[t][0], experts[i])
                current_loss = rej_loss(dat[t][1], current_label, c)
                expert_hyp_losses[i] += current_loss 
        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t + 1))
            count_rej_at_return_rounds.append(count_rej/float(t + 1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbd(c, alpha, experts, dat,return_rounds):
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
            if exp_label(dat[t][0], experts[i]) != -1:  # expert i accepts
                expert_cnt_acc[i] += 1

        for i in range(K):
            exp_prob_acc = expert_cnt_acc[i] / (t+1)
            exp_prob_rej = 1 - exp_prob_acc
            expert_lcbs.append(max(exp_prob_acc - lcb_bound(t+1, t+1, alpha), 0.0) * max((expert_hyp_losses[i] / max(expert_pulls[i],1)) - lcb_bound(t+1, expert_pulls[i], alpha), 0.0) + max(exp_prob_rej - lcb_bound(t+1, t+1, alpha), 0.0) * c) 
        
        #find best arm
        best_arm = expert_lcbs.index(min(expert_lcbs)) 
        
        #update algorithm loss
        expert_label = exp_label(dat[t][0], experts[best_arm])
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        loss_alg += expert_loss

        if expert_label != -1:  # so that we see the label
            for i in range(K):
                expert_pulls[i] += 1
                current_label = exp_hyp_label(dat[t][0], experts[i])
                current_loss = rej_loss(dat[t][1], current_label, c)
                expert_hyp_losses[i] += current_loss 
        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds
       
def ucbh(c, alpha, experts, dat,return_rounds):
    loss_alg_at_return_rounds=[]
    count_rej_at_return_rounds=[]
    enum_return_rounds=0

    K = len(experts)
    T = len(dat)
#    print '\n\n ucbH'
    hyp_expert_avg = {} #only care of emperical average over hyp_experts.
    expert_pulls = [0.0]*K #counts the number of times when r>0
    loss_alg = 0
    count_rej=0

    for t in range(T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t
        acc_exp = {}  
        rej_exp = {}
        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i])
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if i in hyp_expert_avg.keys():
                    acc_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                else:
                    acc_exp[i] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf
            else:
                if i in hyp_expert_avg.keys():
                    rej_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                else:
                    rej_exp[i] = 0.0  # -float("inf")
                    
        #find best arm
        if bool(acc_exp) == True and acc_exp[min(acc_exp, key=acc_exp.get)] < c:
                best_arm = min(acc_exp, key=acc_exp.get)
        else:
            if bool(rej_exp) == True:
                best_arm = min(rej_exp, key=rej_exp.get)
                count_rej+=1
            else:
                best_arm = min(acc_exp, key=acc_exp.get)

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
                        if jj in hyp_expert_avg.keys():
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]
                        else:
                            expert_pulls[jj] += 1
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c)

        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds


def ucbt(c, alpha, experts, dat,return_rounds):
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
            expert_label = exp_label(dat[t][0], experts[i])
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if 1.0-(expert_pulls[i])/(t+1) <= gamma:  # fractin of times r rejected is less than gamma
                    if i in hyp_expert_avg.keys():
                        good_acc_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    else:
                        #i dont think this every happens
                        good_acc_exp[i] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf

                if i in hyp_expert_avg.keys():
                        acc_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                else:
                        acc_exp[i] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf
            else:
                if i in hyp_expert_avg.keys():
                    rej_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                else:
                    rej_exp[i] = 0.0  # -float("inf")
                    

        #find best arm
        if bool(good_acc_exp) == True and good_acc_exp[min(good_acc_exp, key=good_acc_exp.get)] < c:
                best_arm = min(good_acc_exp, key=good_acc_exp.get)
                
        else:
            if bool(rej_exp) == True:
                best_arm = min(rej_exp, key=rej_exp.get)
                count_rej+=1
            else:
                #if no good accepting experts and no rejecting experts pick some random accepting expert
                best_arm = min(acc_exp, key=acc_exp.get)

        
        #update regret
        best_expert_label = save_expert_labels[best_arm]
        expert_loss = rej_loss(dat[t][1], best_expert_label, c) #technically you are calculate loss of best expert twice but meh
        loss_alg += expert_loss
#        print str(dat[t][0])+","+str(best_arm)+","+ str(experts[best_arm])+",   " +str(expert_loss)+","+str(loss_alg) +","+ 'ucbH'
        if best_expert_label != -1:
            # if pulled a nonrej expert update all acc_experts
            for jj in range(len(save_expert_labels)):
                if save_expert_labels[jj] != -1:
                        if jj in hyp_expert_avg.keys():
                            expert_pulls[jj] += 1
                            inv_pull = 1.0 / expert_pulls[jj]
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]
                        else:
                            expert_pulls[jj] += 1
                            hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c)

        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1
    return loss_alg_at_return_rounds,count_rej_at_return_rounds



def ucbvt(c, alpha, experts, dat,return_rounds):

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
                expert_label = exp_label(dat[t][0], experts[i])
                save_expert_labels.append(expert_label)
                if expert_label!=-1:
                    if 1.0-(expert_pulls[i])/(t+1) <= ( 1.0-1.0/float(T) ):  # fractin of times r rejected is less than gamma
                        if i in hyp_expert_avg.keys():
                            good_acc_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                        else:
                            #i dont think this every happens
                            good_acc_exp[i] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf

                    if i in hyp_expert_avg.keys():
                            acc_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    else:
                            acc_exp[i] = 0.0  # -float("inf")  #if you never pulled an arm before the LCB is -inf
                else:
                    if i in hyp_expert_avg.keys():
                        rej_exp[i] = max(hyp_expert_avg[i] - lcb_bound(t,expert_pulls[i],alpha), 0.0)
                    else:
                        rej_exp[i] = 0.0  # -float("inf")


            #find best arm
            if bool(good_acc_exp) == True and good_acc_exp[min(good_acc_exp, key=good_acc_exp.get)] < c:
                    best_arm = min(good_acc_exp, key=good_acc_exp.get)

            else:
                if bool(rej_exp) == True:
                    best_arm = min(rej_exp, key=rej_exp.get)
                    count_rej+=1
                else:
                    best_arm = min(acc_exp, key=acc_exp.get)


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
                            if jj in hyp_expert_avg.keys():
                                expert_pulls[jj] += 1
                                inv_pull = 1.0 / expert_pulls[jj]
                                hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c) * inv_pull + (1.0 - inv_pull) * hyp_expert_avg[jj]
                            else:
                                expert_pulls[jj] += 1
                                hyp_expert_avg[jj] = rej_loss(dat[t][1], save_expert_labels[jj], c)

        # loss_alg_at_return_rounds.append(loss_alg / float(T))    
        # count_rej_at_return_rounds.append(count_rej / float(T))    
        if enum_return_rounds < len(return_rounds) and (t+1)==return_rounds[enum_return_rounds]:
            loss_alg_at_return_rounds.append(loss_alg/float(t+1))
            count_rej_at_return_rounds.append(count_rej/float(t+1))
            enum_return_rounds+=1

    return loss_alg_at_return_rounds,count_rej_at_return_rounds







############# ############# ############# ############# #############  PLOTTING ############# ############# ############# ############# ############# 
def plotting(c,alpha,K,text_file):
#NEED TO IMRPOVE THIS PLOTTING FUNCTION BC IT SUCKS

    NUM_AVG=5
    T_MAX=5000
    avg_regret=[]
    avg_counts=[]
    avg_losses=[]
    for er in range(NUM_AVG):
            experts= create_experts(K, False)
            
            x=range(200,T_MAX,500) 

            loss=[]
            count=[]
            expert_loss=[]

            for p in range(20):
                data=create_data(T_MAX)
                loss_experts=loss_of_every_expert(data,experts,c,x) 
                loss1,countrej1=ucbcc(c,alpha,experts,data,x) #returns values of all needed roudns
                loss2,countrej2=ucbn(c,alpha,experts,data,x)
                loss3,countrej3=ucbh(c,alpha,experts,data,x)
                loss4,countrej4=ucbd(c,alpha,experts,data,x)
#                loss5,countrej5=ucbt(c,alpha,experts,data,x)
                #loss6,countrej6=ucbvt(c,alpha,experts,data,x)
                expert_loss.append(loss_experts)
                loss.append([loss1,loss2,loss3,loss4])#,loss5,loss6])
                count.append([countrej1,countrej2,countrej3,countrej4])#,countrej5,countrej6])
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



    text_file.write('\nPseudo Regret of UCB-type Algorithms for '+str(K)+' arms with c '+str(c))
    text_file.write('; regret UCBC:'+str(avg_regret[0])+'; std UCB:'+str(std_regret[0]))
    text_file.write('; regret UCBN:'+str(avg_regret[1])+'; std UCBN:'+str(std_regret[1]))
    text_file.write('; regret UCBH:'+str(avg_regret[2])+'; std UCBH:'+str(std_regret[2]))
    text_file.write('; regret UCBD:'+str(avg_regret[3])+'; std UCBD:'+str(std_regret[3]))
#    text_file.write('; regret UCBT:'+str(avg_regret[4])+'; std UCBT:'+str(std_regret[4]))
#    text_file.write('; regret UCBVT:'+str(avg_regret[5])+'; std UCBVT:'+str(std_regret[5]))

    text_file.write('; losses UCBC:'+str(avg_losses[0])+'; std UCB:'+str(std_losses[0]))
    text_file.write('; losses UCBN:'+str(avg_losses[1])+'; std UCBN:'+str(std_losses[1]))
    text_file.write('; losses UCBH:'+str(avg_losses[2])+'; std UCBH:'+str(std_losses[2]))
    text_file.write('; losses UCBD:'+str(avg_losses[3])+'; std UCBD:'+str(std_losses[3]))
#    text_file.write('; losses UCBT:'+str(avg_losses[4])+'; std UCBT:'+str(std_losses[4]))
#    text_file.write('; losses UCBVT:'+str(avg_losses[5])+'; std UCBVT:'+str(std_losses[5]))

    text_file.write('; counts UCBC:'+str(avg_counts[0])+'; std UCB:'+str(std_counts[0]))
    text_file.write('; counts UCBN:'+str(avg_counts[1])+'; std UCBN:'+str(std_counts[1]))
    text_file.write('; counts UCBH:'+str(avg_counts[2])+'; std UCBH:'+str(std_counts[2]))
    text_file.write('; counts UCBD:'+str(avg_counts[3])+'; std UCBD:'+str(std_counts[3]))
#    text_file.write('; counts UCBT:'+str(avg_counts[4])+'; std UCBT:'+str(std_counts[4]))
#    text_file.write('; counts UCBVT:'+str(avg_counts[5])+'; std UCBVT:'+str(std_counts[5]))



    fig, ax = plt.subplots()
    ax.errorbar(x, avg_regret[0], yerr=std_regret[0],fmt='r-', label='UCB-C')
    ax.errorbar(x, avg_regret[1], yerr=std_regret[1],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_regret[2], yerr=std_regret[2],fmt='b-', label='UCB-H')
    ax.errorbar(x, avg_regret[3], yerr=std_regret[3],fmt='g-', label='UCB-D')
#    ax.errorbar(x, avg_regret[4], yerr=std_regret[4],fmt='c-', label='UCB-T')
#    ax.errorbar(x, avg_regret[5], yerr=std_regret[5],fmt='y-', label='UCB-VT')
    ax.axhline(y=0.0,c="magenta",linewidth=2,zorder=10)
    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Rounds')
    plt.ylabel('Pseudo-Regret')
    plt.title('Pseudo-Regret of UCB-type Algorithms for '+str(len(experts))+' arms with c '+str(c))
    plt.savefig('./regret_K'+str(len(experts))+'_c'+str(c)+'.png')

    fig, ax = plt.subplots()
    ax.errorbar(x, avg_losses[0], yerr=std_losses[0],fmt='r-', label='UCB-C')
    ax.errorbar(x, avg_losses[1], yerr=std_losses[1],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_losses[2], yerr=std_losses[2],fmt='b-', label='UCB-H')
    ax.errorbar(x, avg_losses[3], yerr=std_losses[3],fmt='g-', label='UCB-D')
#    ax.errorbar(x, avg_losses[4], yerr=std_losses[4],fmt='c-', label='UCB-T')
#    ax.errorbar(x, avg_losses[5], yerr=std_losses[5],fmt='y-', label='UCB-VT')
    ax.axhline(y=0.0,c="magenta",linewidth=2,zorder=10)
    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Rounds')
    plt.ylabel(' Losses')
    plt.title('Losses of UCB-type Algorithms for '+str(len(experts))+' arms with c '+str(c))
    plt.savefig('./losses_K'+str(len(experts))+'_c'+str(c)+'.png')

    fig, ax = plt.subplots()
    ax.errorbar(x, avg_counts[0], yerr=std_counts[0],fmt='r-', label='UCB-C')
    ax.errorbar(x, avg_counts[1], yerr=std_counts[1],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_counts[2], yerr=std_counts[2],fmt='b-', label='UCB-H')
    ax.errorbar(x, avg_counts[3], yerr=std_counts[3],fmt='g-', label='UCB-D')
#    ax.errorbar(x, avg_counts[4], yerr=std_counts[4],fmt='c-', label='UCB-T')
#    ax.errorbar(x, avg_counts[5], yerr=std_counts[5],fmt='y-', label='UCB-VT')
    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Rounds')
    plt.ylabel('Fraction of Rejection')
    plt.title('Fraction of Rejection for '+str(len(experts))+' arms with c '+str(c))
    plt.savefig('./counts_K'+str(len(experts))+'_c'+str(c)+'.png')

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

    alpha=3
    val=int(sys.argv[1])
    K_values=[100,50,20]
    c_values=[0.05,0.2,0.4,0.6,1,2,10]
    if val<=7:
        K=K_values[0]
        c= c_values[int(sys.argv[1])-1]
    elif 7<val and val<=14:
        K=K_values[1]
        c= c_values[int(sys.argv[1])-8]
    elif 14 <val and val <=21:
        K=K_values[2]
        c= c_values[int(sys.argv[1])-15]
    else:
        print 'too high values'

    text_file = open("./Output_" + str(K) + "arms.txt", "w")
    plotting(c,alpha,K,text_file) #last plot point is for T=2000                   
    text_file.close()

