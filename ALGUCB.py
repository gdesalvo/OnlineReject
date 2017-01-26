import random
import math
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
random.seed(1234)

############# ############# ############# ############# #############  HELPER FUNCTIONS ############# ############# ############# ############# ############# 
def create_experts(K):
    #creating the rejecting and hypothesis experts according to threholds
    rej_experts = [random.uniform(0.0, 1.0) for itr in range(K-1)] # reject if rej_experts[i] < data, 
    hyp_experts = [random.uniform(0.0, 0.3) for itr in range(K-1)]  # classification surface is given by threhold rej_expert[i]+hyp_expert[i]
    experts = zip(rej_experts, hyp_experts) #expert format is list of tuple [(rej_threshold1, hyp_threshold1),(rej_threshold2, hyp_threshold2).... ]    
    return experts

def create_data(T):
    #creating data according to gaussian and labels
    x_data = [random.gauss(0.6, 0.1) for itr in range(T)]
    y_labels = [ int(itr >= 0.6)  for itr in x_data]
    data = zip(x_data, y_labels)  #data format is list of tuple [(x1,y1),(x2,y2)....]
    return data


def lcb_bound(current_time, pull, alpha):
    return math.sqrt((alpha / 2)* math.log(current_time) / pull)

def loss_of_best_expert(dat,experts,c):
    return min(loss_of_every_expert(dat, experts, c))

def loss_of_every_expert(dat, experts, c):
    loss_expert = [0] * len(experts)
    for t in range(len(dat)):
        for i in range(len(experts)):
            loss_expert[i] += rej_loss(dat[t][1], exp_label(dat[t][0], experts[i]), c)
    return loss_expert


def rej_loss(true_label, expert_label, c):
    if expert_label == -1:
        return c
    else:
        if true_label == expert_label:
            return 0.0
        else:
            return 1.0
                     

def exp_hyp_label(data, expert):
    if expert[0] + expert[1] <= data:
        expert_label = 1
    else:
        expert_label = 0
    return expert_label

def exp_label(data, expert):
    if expert[0] < data:
        expert_label = exp_hyp_label(data, expert)
    else:
       expert_label = -1
    return expert_label


############# ############# ############# ############# ALGORITHMS ############# ############# ############# ############# ############# ############# 

def ucb(c, alpha, experts, dat): 

    K=len(experts)
    T=len(dat)
    #initialization step
    expert_avg=[]
    loss_alg = 0
    count_rej=0
    for i in range(K):
        expert_label=exp_label(dat[i][0], experts[i])
        if expert_label==-1:
            count_rej+=1
        exp_loss = rej_loss(dat[i][1], expert_label, c)
        expert_avg.append(exp_loss)
        loss_alg += exp_loss
    expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

    for t in range(K, T):
        #find best arm
        lcb_list = [expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha) for i in range(K)] #soinefficient
        best_arm = lcb_list.index(min(lcb_list)) 

        expert_pulls[best_arm] += 1 #update number of times arm is pulled

        #update loss of best arm average
        inv_pull = 1.0 / expert_pulls[best_arm]
        expert_loss = rej_loss(dat[t][1], exp_label(dat[t][0], experts[best_arm]), c)
        expert_avg[best_arm] = expert_loss * inv_pull + (1-inv_pull) * expert_avg[best_arm]
        if exp_label(dat[t][0],experts[best_arm]) == -1:
            count_rej+=1

        #update regret
        loss_alg += expert_loss

    return loss_alg / float(T) , count_rej/float(T)


def ucbn(c, alpha, experts, dat):
    K = len(experts)
    T = len(dat)

    expert_avg = []
    loss_alg = 0
    count_rej=0
    #initialization step
    for i in range(K):
        expert_label=exp_label(dat[i][0], experts[i])
        if expert_label==-1:
            count_rej+=1
        exp_loss = rej_loss(dat[i][1], expert_label, c)
        expert_avg.append(exp_loss)
        loss_alg += exp_loss
    expert_pulls = [1.0] * K #number of times expert is observed! 

    for t in range(K, T):
        #find best arm
        lcb_list=[expert_avg[i] - lcb_bound(t, expert_pulls[i], alpha) for i in range(K)] #soinefficient
        best_arm = lcb_list.index(min(lcb_list)) 
        
        #update regret
        expert_label = exp_label(dat[t][0], experts[best_arm])
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
#        print best_arm,expert_loss
        loss_alg += expert_loss
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
    return loss_alg / float(T),count_rej /float(T) 


def ucbd(c, alpha, experts, dat):
    K = len(experts)
    T = len(dat)

    expert_cnt_acc = [0.0] * K
    expert_pulls = [0.0] * K
    expert_hyp_losses = [0.0] * K
    loss_alg = 0
    count_rej=0

    #initialization step
    for i in range(K):
        expert_label = exp_hyp_label(dat[i][0], experts[i])  # explicitly accept
        if expert_label==-1:
            count_rej+=1
        for j in range(K):
            if exp_label(dat[i][0], experts[j]) != -1:
                expert_cnt_acc[j] += 1
            expert_pulls[j] += 1  # always increment upon accept
        expert_hyp_loss = rej_loss(dat[i][1], expert_label, c)
        expert_hyp_losses[i] += expert_hyp_loss
        loss_alg += expert_hyp_loss

    for t in range(K, T):

        # aggregate lcbs
        expert_lcbs = []
        for i in range(K):
            exp_prob_acc = expert_cnt_acc[i] / t
            exp_prob_rej = 1 - exp_prob_acc
            expert_lcbs.append((exp_prob_acc - lcb_bound(t, t, alpha)) * ((expert_hyp_losses[i] / expert_pulls[i]) - lcb_bound(t, expert_pulls[i], alpha)) + (exp_prob_rej - lcb_bound(t, t, alpha)) * c) 
        
        #find best arm
        best_arm = expert_lcbs.index(min(expert_lcbs)) 
        
        #update algorithm loss
        expert_label = exp_label(dat[t][0], experts[best_arm])
        expert_loss = rej_loss(dat[t][1], expert_label, c) 
        if expert_label == -1:
            count_rej+=1
        loss_alg += expert_loss

        # update slacks and empirical averages
        for i in range(K):
            if exp_label(dat[t][0], experts[i]) != -1:  # expert i accepts
                expert_cnt_acc[i] += 1
        if expert_label != -1:  # so that we see the label
            for i in range(K):
                expert_pulls[i] += 1
                current_label = exp_hyp_label(dat[t][0], experts[i])
                current_loss = rej_loss(dat[t][1], current_label, c)
                expert_hyp_losses[i] += current_loss 
    return loss_alg/ float(T) , count_rej/float(T)

       
def ucbh(c, alpha, experts, dat):
    K = len(experts)
    T = len(dat)

    hyp_expert_avg = {} #only care of emperical average over hyp_experts.
    expert_pulls = [] #counts the number of times when r>0
    loss_alg = 0
    count_rej=0
    #initialization step
    for i in range(K):
        expert_label = exp_label(dat[i][0], experts[i])
        if expert_label != -1:
            exp_loss = rej_loss(dat[i][1], expert_label, c) 
            hyp_expert_avg[str(i)] = exp_loss  #prob should define zero/one loss and use it here
            loss_alg += exp_loss
            expert_pulls.append(1)
        else:
            expert_pulls.append(0)
            loss_alg += c
            count_rej+=1
    
    for t in range(K, T):
        #use dictionary so keep track of which are accepting and rejecting experts at time t
        acc_exp = {}  
        rej_exp = {}

        save_expert_labels=[]
        for i in range(K):
            #separate rejecting and accepting experts according to their label (not really kosher but essence the same)
            expert_label = exp_label(dat[t][0], experts[i])
            save_expert_labels.append(expert_label)
            if expert_label!=-1:
                if str(i) in hyp_expert_avg.keys():
                    acc_exp[str(i)] = hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha)
                else:
                    acc_exp[str(i)] = -float("inf")  #if you never pulled an arm before the LCB is -inf
            else:
                if str(i) in hyp_expert_avg.keys():
                    rej_exp[str(i)] = hyp_expert_avg[str(i)] - lcb_bound(t,expert_pulls[i],alpha)
                else:
                    rej_exp[str(i)] = -float("inf")
                    

        #find best arm
        if bool(acc_exp) == True and acc_exp[min(acc_exp, key=acc_exp.get)] < c:
                best_arm = int(min(acc_exp, key=acc_exp.get))
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


    return loss_alg / float(T), count_rej/float(T)

############# ############# ############# ############# #############  PLOTTING ############# ############# ############# ############# ############# 
def plotting(c,alpha,K,text_file):
#NEED TO IMRPOVE THIS PLOTTING FUNCTION BC IT SUCKS
    NUM_AVG=4
    avg_regret=[]
    avg_counts=[]
    avg_losses=[]
    for er in range(NUM_AVG):
                #for plotting
            loss=[]
            count_rejection=[]
            expert_loss=[]
            x=range(200,5000,200) #maybe use T instead?
            for rounds in x:
                loss_at_t=[]
                count_at_t=[]
                expert_loss_at_t=[]
                experts= create_experts(K)

                for p in range(10):
                        data=create_data(rounds)
                        loss_experts=loss_of_every_expert(data,experts,c)                
                        loss1,countrej1=ucb(c,alpha,experts,data)
                        loss2,countrej2=ucbn(c,alpha,experts,data)
                        loss3,countrej3=ucbh(c,alpha,experts,data)
                        loss4,countrej4=ucbd(c,alpha,experts,data)
                        expert_loss_at_t.append(loss_experts)
                        loss_at_t.append([loss1,loss2,loss3,loss4 ])
                        count_at_t.append([countrej1,countrej2,countrej3,countrej4])

                        
                loss_at_t=np.array(loss_at_t)
                loss.append(np.mean(loss_at_t,axis=0 ))
                count_rejection.append(np.mean(np.array(count_at_t),axis=0))
                expert_loss.append(np.mean(np.array(expert_loss_at_t)/float(rounds),axis=0))



            loss=np.array(loss)
            count_rejection=np.array(count_rejection)

            expert_loss=np.array(expert_loss)
            best_expert_loss=np.amin(expert_loss,axis=1)

            regret=loss-np.expand_dims(best_expert_loss,axis=1)
            
            avg_regret.append(regret)
            avg_losses.append(loss)
            avg_counts.append(count_rejection)
    
    std_regret=np.std(np.array(avg_regret),axis=0)
    avg_regret=np.mean(np.array(avg_regret),axis=0)
    std_losses=np.std(np.array(avg_losses),axis=0)
    avg_losses=np.mean(np.array(avg_losses),axis=0)
    std_counts=np.std(np.array(avg_counts),axis=0)
    avg_counts=np.mean(np.array(avg_counts),axis=0)



    text_file.write('\nPseudo Regret of UCB-type Algorithms for '+str(K)+' arms with c '+str(c))
    text_file.write('; regret UCB:'+str(avg_regret[:,0])+'; std UCB:'+str(std_regret[:,0]))
    text_file.write('; regret UCBN:'+str(avg_regret[:,1])+'; std UCBN:'+str(std_regret[:,1]))
    text_file.write('; regret UCBH:'+str(avg_regret[:,2])+'; std UCBH:'+str(std_regret[:,2]))
    text_file.write('; regret UCBD:'+str(avg_regret[:,3])+'; std UCBD:'+str(std_regret[:,3]))
    text_file.write('; losses UCB:'+str(avg_losses[:,0])+'; std UCB:'+str(std_losses[:,0]))
    text_file.write('; losses UCBN:'+str(avg_losses[:,1])+'; std UCBN:'+str(std_losses[:,1]))
    text_file.write('; losses UCBH:'+str(avg_losses[:,2])+'; std UCBH:'+str(std_losses[:,2]))
    text_file.write('; losses UCBD:'+str(avg_losses[:,3])+'; std UCBD:'+str(std_losses[:,3]))
    text_file.write('; counts UCB:'+str(avg_counts[:,0])+'; std UCB:'+str(std_counts[:,0]))
    text_file.write('; counts UCBN:'+str(avg_counts[:,1])+'; std UCBN:'+str(std_counts[:,1]))
    text_file.write('; counts UCBH:'+str(avg_counts[:,2])+'; std UCBH:'+str(std_counts[:,2]))
    text_file.write('; counts UCBD:'+str(avg_counts[:,3])+'; std UCBD:'+str(std_counts[:,3]))



    fig, ax = plt.subplots()
    ax.errorbar(x, avg_regret[:,0], yerr=std_regret[:,0],fmt='r-', label='UCB')
    ax.errorbar(x, avg_regret[:,1], yerr=std_regret[:,1],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_regret[:,2], yerr=std_regret[:,2],fmt='b-', label='UCB-H')
    ax.errorbar(x, avg_regret[:,3], yerr=std_regret[:,3],fmt='g-', label='UCB-D')
    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Rounds')
    plt.ylabel('Pseudo-Regret')
    plt.title('Pseudo-Regret of UCB-type Algorithms for '+str(K)+' arms with c '+str(c))
    plt.savefig('./figures/regret_K'+str(K)+'_c'+str(c)+'.png')

    fig, ax = plt.subplots()
    ax.errorbar(x, avg_losses[:,0], yerr=std_losses[:,0],fmt='r-', label='UCB')
    ax.errorbar(x, avg_losses[:,1], yerr=std_losses[:,1],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_losses[:,2], yerr=std_losses[:,2],fmt='b-', label='UCB-H')
    ax.errorbar(x, avg_losses[:,3], yerr=std_losses[:,3],fmt='g-', label='UCB-D')
    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Rounds')
    plt.ylabel(' Losses')
    plt.title('Losses of UCB-type Algorithms for '+str(K)+' arms with c '+str(c))
    plt.savefig('./figures/losses_K'+str(K)+'_c'+str(c)+'.png')

    fig, ax = plt.subplots()
    ax.errorbar(x, avg_counts[:,0], yerr=std_counts[:,0],fmt='r-', label='UCB')
    ax.errorbar(x, avg_counts[:,1], yerr=std_counts[:,1],fmt='k-', label='UCB-N')
    ax.errorbar(x, avg_counts[:,2], yerr=std_counts[:,2],fmt='b-', label='UCB-H')
    ax.errorbar(x, avg_counts[:,3], yerr=std_counts[:,3],fmt='g-', label='UCB-D')
    legend = ax.legend(loc='upper right', shadow=True)
    plt.xlabel('Rounds')
    plt.ylabel('Expected Counts')
    plt.title('Expected Counts of UCB-type Algorithms for '+str(K)+' arms with c '+str(c))
    plt.savefig('./figures/counts_K'+str(K)+'_c'+str(c)+'.png')

############# ############# ############# ############# #############  MAIN ############# ############# ############# ############# ############# 
if __name__ == "__main__":

    #parameters
    c = 0.45
    if len(sys.argv) > 4:
        c = float(sys.argv[4])
    alpha = 3
    K = int(sys.argv[1])
    T = int(sys.argv[2])
    text_file = open("./figures/Output_" + str(K) + "arms.txt", "w")
    #All algorithms should share same data and same experts. 
#    experts, data = create_experts_and_data(K, T)
#    avgloss_best = loss_of_best_expert(data, experts, c) / T
#    avgloss_ucb = ucb(c, alpha, experts, data)
#    avgloss_ucbn = ucbn(c, alpha, experts, data)
#    avgloss_ucbh = ucbh(c, alpha, experts, data)
#    avgloss_ucbd = ucbd(c, alpha, experts, data)
#    reg1 = avgloss_ucb - avgloss_best
#    reg2 = avgloss_ucbn - avgloss_best
#    reg3 = avgloss_ucbh - avgloss_best
#    reg4 = avgloss_ucbd - avgloss_best

#    print "loss best arm " + str(avgloss_best)
#    print "loss of UCB " + str(avgloss_ucb) + ", loss of UCB-N " + str(avgloss_ucbn) + ", loss of UCB-H " + str(avgloss_ucbh) + ", loss of UCB-D " + str(avgloss_ucbd)
#    print "regret of UCB " + str(reg1) + ", regret of UCB-N " + str(reg2) + ", regret of UCB-H " + str(reg3) + ", regret of UCB-D " + str(reg4)
    
    if int(sys.argv[3]) == 1:
#        c_values=[0.05, 0.4]
        c_values=[0.05,0.2,0.4,0.6,1,2,10]
        for c in c_values:
            print 'workin on c'+str(c)
            plotting(c,alpha,K,text_file) #last plot point is for T=2000
            
    text_file.close()
