import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_path='../data/final_result/amazon/tip_num_amp3_n.csv'
no_obs_data_path='../data/final_result/amazon/tip_num_non_obs_amp3_n.csv'
output='../../output/final_result/amazon/scenario_amp3_n'

data=pd.read_csv(data_path,header=None)
no_obs_data=pd.read_csv(no_obs_data_path,header=None)

group_data=data.groupby([0,1,2])
accuracy_mean=np.zeros((3,5,6))
accuracy_var=np.zeros((3,5,6))

for i,group in group_data:
    group_num=group.iat[0,0]

    group_obs_ind=(group.iat[0,2]==0.01)*0
    group_obs_ind+=(group.iat[0,1]==0.025)*1
    group_obs_ind+=(group.iat[0,1]==0.05)*2
    group_obs_ind+=(group.iat[0,1]==0.1)*3
    group_obs_ind+=(group.iat[0,1]==0.2)*4

    group_num_ind=(group.iat[0,2]==10)*0
    group_num_ind+=(group.iat[0,2]==20)*1
    group_num_ind+=(group.iat[0,2]==40)*2
    group_num_ind+=(group.iat[0,2]==80)*3
    group_num_ind+=(group.iat[0,2]==100)*4
    group_num_ind+=(group.iat[0,2]==200)*5

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('number of tipping particle')
    ax.hist(group.iloc[:,3],range=(0,1000),color='b')
    ax.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3],range=(0,1000),color=(1, 0, 0, 0.2))
    mean=group.iloc[:,3].mean()
    print(mean)
    var=group.iloc[:,3].var()
    plt.title(
        'group:{},v_obs:{},num:{}'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.savefig(
        output+'/group_{}_{}_{}.png'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.close()
    plt.clf()
    accuracy_mean[group_num-1,group_obs_ind,group_num_ind]+=mean
    accuracy_var[group_num-1,group_obs_ind,group_num_ind]+=var

plt.figure(1)

for i,group in group_data:
    group_num=group.iat[0,0]
    if group_num==1:

        group_obs_ind=(group.iat[0,2]==0.01)*0
        group_obs_ind+=(group.iat[0,1]==0.025)*1
        group_obs_ind+=(group.iat[0,1]==0.05)*2
        group_obs_ind+=(group.iat[0,1]==0.1)*3
        group_obs_ind+=(group.iat[0,1]==0.2)*4

        group_num_ind=(group.iat[0,2]==10)*0
        group_num_ind+=(group.iat[0,2]==20)*1
        group_num_ind+=(group.iat[0,2]==40)*2
        group_num_ind+=(group.iat[0,2]==80)*3
        group_num_ind+=(group.iat[0,2]==100)*4
        group_num_ind+=(group.iat[0,2]==200)*5


        plt.subplot(5,6,group_obs_ind*6+group_num_ind+1)
        if (group_obs_ind==4)&(group_num_ind!=0):
            plt.xticks([0,1])
            plt.yticks([])
        if (group_num_ind==0)&(group_obs_ind!=4):
            plt.xticks([])
            plt.yticks([0,50,100])
        if (group_obs_ind!=4)&(group_num_ind!=0):
            plt.xticks([0,1])
            plt.yticks([0,50,100])
        if (group_obs_ind!=4)&(group_num_ind!=0):
            plt.xticks([])
            plt.yticks([])
        plt.hist(group.iloc[:,3]/1000,range=(0,1),color='b')
        plt.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3]/1000,range=(0,1),color=(1, 0, 0, 0.2))

plt.savefig(
    output+'/all_hist_1.png')
plt.close()
plt.clf()

plt.figure(1)

for i,group in group_data:
    group_num=group.iat[0,0]
    if group_num==2:

        group_obs_ind=(group.iat[0,2]==0.01)*0
        group_obs_ind=(group.iat[0,1]==0.025)*1
        group_obs_ind+=(group.iat[0,1]==0.05)*2
        group_obs_ind+=(group.iat[0,1]==0.1)*3
        group_obs_ind+=(group.iat[0,1]==0.2)*4

        group_num_ind=(group.iat[0,2]==10)*0
        group_num_ind+=(group.iat[0,2]==20)*1
        group_num_ind+=(group.iat[0,2]==40)*2
        group_num_ind+=(group.iat[0,2]==80)*3
        group_num_ind+=(group.iat[0,2]==100)*4
        group_num_ind+=(group.iat[0,2]==200)*5


        plt.subplot(5,6,group_obs_ind*6+group_num_ind+1)
        if (group_obs_ind==4)&(group_num_ind!=0):
            plt.xticks([0,1])
            plt.yticks([])
        if (group_num_ind==0)&(group_obs_ind!=4):
            plt.xticks([])
            plt.yticks([0,50,100])
        if (group_obs_ind!=4)&(group_num_ind!=0):
            plt.xticks([0,1])
            plt.yticks([0,50,100])
        if (group_obs_ind!=4)&(group_num_ind!=0):
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

for i,group in group_data:
    group_num=group.iat[0,0]
    if group_num==3:

        group_obs_ind=(group.iat[0,2]==0.01)*0
        group_obs_ind=(group.iat[0,1]==0.025)*1
        group_obs_ind+=(group.iat[0,1]==0.05)*2
        group_obs_ind+=(group.iat[0,1]==0.1)*3
        group_obs_ind+=(group.iat[0,1]==0.2)*4

        group_num_ind=(group.iat[0,2]==10)*0
        group_num_ind+=(group.iat[0,2]==20)*1
        group_num_ind+=(group.iat[0,2]==40)*2
        group_num_ind+=(group.iat[0,2]==80)*3
        group_num_ind+=(group.iat[0,2]==100)*4
        group_num_ind+=(group.iat[0,2]==200)*5


        plt.subplot(5,6,group_obs_ind*6+group_num_ind+1)
        if (group_obs_ind==4)&(group_num_ind!=0):
            plt.xticks([0,1])
            plt.yticks([])
        if (group_num_ind==0)&(group_obs_ind!=4):
            plt.xticks([])
            plt.yticks([0,50,100])
        if (group_obs_ind!=4)&(group_num_ind!=0):
            plt.xticks([0,1])
            plt.yticks([0,50,100])
        if (group_obs_ind!=4)&(group_num_ind!=0):
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
    group_num=group.iat[0,0]
    mean=group.iloc[:,3].mean()
    var=group.iloc[:,3].var()
    no_obs_accuracy_mean[group_num-1]+=mean
    no_obs_accuracy_var[group_num-1]+=var
##group1 mean
plt.figure()
sns.heatmap(
    (accuracy_mean[0,:,:]-no_obs_accuracy_mean[0])/1000,
    vmin=0.5,
    vmax=-0.5,
    cmap='bwr',
    yticklabels=[0.01,0.025,0.05,0.1,0.2],
    xticklabels=[10,20,40,80,100,200])

plt.savefig(output+'/group1_accuracy_mean_map.png')
plt.close()
plt.clf()
##group1 mvar
plt.figure()
sns.heatmap(
    accuracy_var[0,:,:]/1000000,
    cmap='gist_gray',
    yticklabels=[0.01,0.025,0.05,0.1,0.2],
    xticklabels=[10,20,40,80,100,200])

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
    yticklabels=[0.01,0.025,0.05,0.1,0.2],
    xticklabels=[10,20,40,80,100,200])

plt.savefig(output+'/group2_accuracy_mean_map.png')
plt.close()
plt.clf()
##group2 var
plt.figure()
sns.heatmap(
    accuracy_var[1,:,:]/1000000,
    cmap='gist_gray',
    yticklabels=[0.01,0.025,0.05,0.1,0.2],
    xticklabels=[10,20,40,80,100,200])

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
    yticklabels=[0.01,0.025,0.05,0.1,0.2],
    xticklabels=[10,20,40,80,100,200])

plt.savefig(output+'/group3_accuracy_mean_map.png')
plt.close()
plt.clf()
##group3 var
plt.figure()
sns.heatmap(
    accuracy_var[2,:,:]/1000000,
    cmap='gist_gray',
    yticklabels=[0.01,0.025,0.05,0.1,0.2],
    xticklabels=[10,20,40,80,100,200])

plt.savefig(output+'/group3_accuracy_var_map.png')
plt.close()
plt.clf()