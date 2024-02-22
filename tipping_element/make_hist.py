import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_path='../data/final_result/amoc/tip_num_freq.csv'
no_obs_data_path='../data/final_result/amoc/tip_num_non_obs2.csv'
data_200_path='../data/final_result/amoc/tip_num2.csv'
output='../../output/final_result/amoc/scenario_freq'

data=pd.read_csv(data_path,header=None)
no_obs_data=pd.read_csv(no_obs_data_path,header=None)
data_200=pd.read_csv(data_200_path,header=None)

group_data=data.groupby([0,1,2])
accuracy_mean=np.zeros((3,4,4)) #amoc
accuracy_var=np.zeros((3,4,4)) #amoc

#accuracy_mean=np.zeros((3,3,4)) #amazon
#accuracy_var=np.zeros((3,3,4)) #amazon

for i,group in group_data:
    group_fs=group.iat[0,0]

    group_obs_ind=(group.iat[0,1]==0.01)*0
    group_obs_ind=(group.iat[0,1]==0.025)*1
    group_obs_ind+=(group.iat[0,1]==0.05)*2
    group_obs_ind+=(group.iat[0,1]==0.1)*3

    group_fs_ind=(group.iat[0,2]==70)*1
    group_fs_ind+=(group.iat[0,2]==80)*2
    group_fs_ind+=(group.iat[0,2]==90)*3

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('number of tipping particle')
    ax.hist(group.iloc[:,3]/1000,range=(0,1),color='b')
    ax.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))
    mean=group.iloc[:,3].mean()
    print(mean)
    var=group.iloc[:,3].var()
    plt.title(
        'group:{},v_obs:{},fs:{}'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.savefig(
        output+'/group_{}_{}_{}.png'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.close()
    plt.clf()
    accuracy_mean[group_fs-1,group_obs_ind,group_fs_ind-1]+=mean
    accuracy_var[group_fs-1,group_obs_ind,group_fs_ind-1]+=var

plt.figure(1)
data_200_group=data_200[(data_200.iloc[:,0]==1)&(data_200.iloc[:,2]==100)]
#for i,acc in enumerate([0.01,0.025,0.05]): #amazon
for i,acc in enumerate([0.01,0.025,0.05,0.1]): #amoc
    #plt.subplot(3,4,(i+1)*4) #amazon
    plt.subplot(4,4,(i+1)*4) #amoc
    plt.hist(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3]/1000,range=(0,1),color='b')
    plt.hist(no_obs_data[no_obs_data[0]==1].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))
    #if i!=2:#amazon
    if i!=3:#amoc
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks([0,1])
        plt.yticks([])
    accuracy_mean[0,i,3]+=np.mean(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3])
    accuracy_var[0,i,3]+=np.var(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3])

for i,group in group_data:
    group_fs=group.iat[0,0]
    if group_fs==1:

        group_obs_ind=(group.iat[0,1]==0.01)*0
        group_obs_ind+=(group.iat[0,1]==0.025)*1
        group_obs_ind+=(group.iat[0,1]==0.05)*2
        group_obs_ind+=(group.iat[0,1]==0.1)*3 #amoc

        group_fs_ind=(group.iat[0,2]==90)*2
        group_fs_ind+=(group.iat[0,2]==80)*1
        group_fs_ind+=(group.iat[0,2]==70)*0


        #plt.subplot(3,4,group_obs_ind*4+group_fs_ind+1) #amazon
        plt.subplot(4,4,group_obs_ind*4+group_fs_ind+1) #amoc
        #if (group_obs_ind==2)&(group_fs_ind!=0): #amazon
        if (group_obs_ind==3)&(group_fs_ind!=0):#amoc
            plt.xticks([0,1])
            plt.yticks([])
        #if (group_fs_ind==0)&(group_obs_ind!=2):#amazon
        if (group_fs_ind==0)&(group_obs_ind!=3):#amoc
            plt.xticks([])
            plt.yticks([0,50,100])
        #if (group_obs_ind==2)&(group_fs_ind==0): #amazon
        if (group_obs_ind==3)&(group_fs_ind==0): #amoc
            plt.xticks([0,1])
            plt.yticks([0,50,100])
        #if (group_obs_ind!=2)&(group_fs_ind!=0): #amazon
        if (group_obs_ind!=3)&(group_fs_ind!=0): #amoc
            plt.xticks([])
            plt.yticks([])
        plt.hist(group.iloc[:,3]/1000,range=(0,1),color='b')
        plt.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))

plt.savefig(
    output+'/all_hist_1.png')
plt.close()
plt.clf()

plt.figure(1)
data_200_group=data_200[(data_200.iloc[:,0]==2)&(data_200.iloc[:,2]==100)]
#for i,acc in enumerate([0.01,0.025,0.05]): #amazon
for i,acc in enumerate([0.01,0.025,0.05,0.1]): #amoc
    #plt.subplot(3,4,(i+1)*4) #amazon
    plt.subplot(4,4,(i+1)*4) #amoc
    #if i!=2: #amazon
    if i!=3: #amoc
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks([0,1])
        plt.yticks([])
    plt.hist(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3]/1000,range=(0,1),color='b')
    plt.hist(no_obs_data[no_obs_data[0]==2].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))
    accuracy_mean[1,i,3]+=np.mean(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3])
    accuracy_var[1,i,3]+=np.var(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3])

for i,group in group_data:
    group_fs=group.iat[0,0]
    if group_fs==2:

        group_obs_ind=(group.iat[0,1]==0.01)*0
        group_obs_ind+=(group.iat[0,1]==0.025)*1
        group_obs_ind+=(group.iat[0,1]==0.05)*2
        group_obs_ind+=(group.iat[0,1]==0.1)*3 #amoc

        group_fs_ind=(group.iat[0,2]==90)*2
        group_fs_ind+=(group.iat[0,2]==80)*1
        group_fs_ind+=(group.iat[0,2]==70)*0


        #plt.subplot(3,4,group_obs_ind*4+group_fs_ind+1) #amazon
        plt.subplot(4,4,group_obs_ind*4+group_fs_ind+1) #amoc
        #if (group_obs_ind==2)&(group_fs_ind!=0): #amazon
        if (group_obs_ind==3)&(group_fs_ind!=0):#amoc
            plt.xticks([0,1])
            plt.yticks([])
        #if (group_fs_ind==0)&(group_obs_ind!=2):#amazon
        if (group_fs_ind==0)&(group_obs_ind!=3):#amoc
            plt.xticks([])
            plt.yticks([0,50,100])
        #if (group_obs_ind==2)&(group_fs_ind==0): #amazon
        if (group_obs_ind==3)&(group_fs_ind==0): #amoc
            plt.xticks([0,1])
            plt.yticks([0,50,100])
        #if (group_obs_ind!=2)&(group_fs_ind!=0): #amazon
        if (group_obs_ind!=3)&(group_fs_ind!=0): #amoc
            plt.xticks([])
            plt.yticks([])
        plt.hist(group.iloc[:,3]/1000,range=(0,1),color='b')
        plt.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))

plt.savefig(
    output+'/all_hist_2.png')
plt.close()
plt.clf()

plt.figure(1)

plt.figure(1)
data_200_group=data_200[(data_200.iloc[:,0]==1)&(data_200.iloc[:,2]==100)]
#for i,acc in enumerate([0.01,0.025,0.05]): #amazon
for i,acc in enumerate([0.01,0.025,0.05,0.1]): #amoc
    #plt.subplot(3,4,(i+1)*4)#amazon
    plt.subplot(4,4,(i+1)*4) #amoc
    #if i!=2: #amazon
    if i!=3: #amoc
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks([0,1])
        plt.yticks([])
    plt.hist(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3]/1000,range=(0,1),color='b')
    plt.hist(no_obs_data[no_obs_data[0]==3].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))
    accuracy_mean[2,i,3]+=np.mean(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3])
    accuracy_var[2,i,3]+=np.var(data_200_group[data_200_group.iloc[:,1]==acc].iloc[:,3])

for i,group in group_data:
    group_fs=group.iat[0,0]
    if group_fs==3:

        group_obs_ind=(group.iat[0,1]==0.01)*0
        group_obs_ind+=(group.iat[0,1]==0.025)*1
        group_obs_ind+=(group.iat[0,1]==0.05)*2
        group_obs_ind+=(group.iat[0,1]==0.1)*3 #amoc

        group_fs_ind=(group.iat[0,2]==90)*2
        group_fs_ind+=(group.iat[0,2]==80)*1
        group_fs_ind+=(group.iat[0,2]==70)*0


        #plt.subplot(3,4,group_obs_ind*4+group_fs_ind+1) #amazon
        plt.subplot(4,4,group_obs_ind*4+group_fs_ind+1) #amoc
        #if (group_obs_ind==2)&(group_fs_ind!=0): #amazon
        if (group_obs_ind==3)&(group_fs_ind!=0):#amoc
            plt.xticks([0,1])
            plt.yticks([])
        #if (group_fs_ind==0)&(group_obs_ind!=2):#amazon
        if (group_fs_ind==0)&(group_obs_ind!=3):#amoc
            plt.xticks([])
            plt.yticks([0,50,100])
        #if (group_obs_ind==2)&(group_fs_ind==0): #amazon
        if (group_obs_ind==3)&(group_fs_ind==0): #amoc
            plt.xticks([0,1])
            plt.yticks([0,50,100])
        #if (group_obs_ind!=2)&(group_fs_ind!=0): #amazon
        if (group_obs_ind!=3)&(group_fs_ind!=0): #amoc
            plt.xticks([])
            plt.yticks([])
        plt.hist(group.iloc[:,3]/1000,range=(0,1),color='b')
        plt.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))

plt.savefig(
    output+'/all_hist_3.png')
plt.close()
plt.clf()

##without any observations
no_obs_accuracy_mean=np.zeros((3,))
no_obs_accuracy_var=np.zeros((3,))

no_obs_group_data=no_obs_data.groupby([0])
for i,group in no_obs_group_data:
    group_fs=group.iat[0,0]
    mean=group.iloc[:,3].mean()
    var=group.iloc[:,3].var()
    no_obs_accuracy_mean[group_fs-1]+=mean
    no_obs_accuracy_var[group_fs-1]+=var
##group1 mean
plt.figure()
sns.heatmap(
    (accuracy_mean[0,:,:]-no_obs_accuracy_mean[0])/1000,
    vmin=0.5,
    vmax=-0.5,
    cmap='bwr',
    #yticklabels=[0.01,0.025,0.05], #amazon
    yticklabels=[0.01,0.025,0.05,0.1],#amoc
    xticklabels=[70,80,90,100])

plt.savefig(output+'/group1_accuracy_mean_map.png')
plt.close()
plt.clf()
##group1 mvar
plt.figure()
sns.heatmap(
    accuracy_var[0,:,:]/1000000,
    cmap='gist_gray',
    #yticklabels=[0.01,0.025,0.05], #amazon
    yticklabels=[0.01,0.025,0.05,0.1],#amoc
    xticklabels=[70,80,90,100])

plt.savefig(output+'/group1_accuracy_var_map.png')
plt.close()
plt.clf()
##group2 mean
plt.figure()
sns.heatmap(
    (accuracy_mean[1,:,:]-no_obs_accuracy_mean[1])/1000,
    cmap='bwr',
    vmin=-0.5,
    vmax=0.5,
    #yticklabels=[0.01,0.025,0.05], #amazon
    yticklabels=[0.01,0.025,0.05,0.1],#amoc
    xticklabels=[70,80,90,100])

plt.savefig(output+'/group2_accuracy_mean_map.png')
plt.close()
plt.clf()
##group2 var
plt.figure()
sns.heatmap(
    accuracy_var[1,:,:]/1000000,
    cmap='gist_gray',
    #yticklabels=[0.01,0.025,0.05], #amazon
    yticklabels=[0.01,0.025,0.05,0.1],#amoc
    xticklabels=[70,80,90,100])

plt.savefig(output+'/group2_accuracy_var_map.png')
plt.close()
plt.clf()
##group3 mean
plt.figure()
sns.heatmap(
    ((-1)*accuracy_mean[2,:,:]+no_obs_accuracy_mean[2])/1000,
    cmap='bwr',
    vmin=-0.5,
    vmax=0.5,
    #yticklabels=[0.01,0.025,0.05], #amazon
    yticklabels=[0.01,0.025,0.05,0.1],#amoc
    xticklabels=[70,80,90,100])

plt.savefig(output+'/group3_accuracy_mean_map.png')
plt.close()
plt.clf()
##group3 var
plt.figure()
sns.heatmap(
    accuracy_var[2,:,:]/1000000,
    cmap='gist_gray',
    #yticklabels=[0.01,0.025,0.05], #amazon
    yticklabels=[0.01,0.025,0.05,0.1],#amoc
    xticklabels=[70,80,90,100])

plt.savefig(output+'/group3_accuracy_var_map.png')
plt.close()
plt.clf()