import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_path='../data/amoc/tip_num.csv'
no_obs_data_path='../data/amoc/tip_num_non_obs.csv'

data=pd.read_csv(data_path,header=None)
no_obs_data=pd.read_csv(no_obs_data_path,header=None)

group_data=data.groupby([0,1,2])
accuracy_mean=np.zeros((3,4,6))
accuracy_var=np.zeros((3,4,6))

for i,group in group_data:
    group_num=group.iat[0,0]

    group_obs_ind=(group.iat[0,1]==0.025)*0
    group_obs_ind+=(group.iat[0,1]==0.05)*1
    group_obs_ind+=(group.iat[0,1]==0.1)*2
    group_obs_ind+=(group.iat[0,1]==0.2)*3

    group_fs_ind=(group.iat[0,2]==201)*0
    group_fs_ind+=(group.iat[0,2]==101)*1
    group_fs_ind+=(group.iat[0,2]==81)*2
    group_fs_ind+=(group.iat[0,2]==41)*3
    group_fs_ind+=(group.iat[0,2]==21)*4
    group_fs_ind+=(group.iat[0,2]==11)*5

    group.iloc[:,2]=2000/(group.iloc[:,2]-1)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('number of tipping particle')
    ax.hist(group.iloc[:,3],range=(0,1000),color='b')
    ax.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3],range=(0,1000),color=(1, 0, 0, 0.2))
    mean=group.iloc[:,3].mean()
    print(mean)
    var=group.iloc[:,3].var()
    plt.title(
        'group:{},v_obs:{},fs:{}'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.savefig(
        '../../output/amoc/scenario/group_{}_{}_{}.png'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.close()
    plt.clf()

    accuracy_mean[group_num-1,group_obs_ind,group_fs_ind]+=mean
    accuracy_var[group_num-1,group_obs_ind,group_fs_ind]+=var

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
    yticklabels=[0.025,0.05,0.1,0.2],
    xticklabels=[11,21,41,81,101,201])

plt.savefig('../../output/amoc/scenario/group1_accuracy_mean_map.png')
plt.close()
plt.clf()
##group1 mvar
plt.figure()
sns.heatmap(
    accuracy_var[0,:,:]/1000000,
    cmap='gist_gray',
    yticklabels=[0.025,0.05,0.1,0.2],
    xticklabels=[11,21,41,81,101,201])

plt.savefig('../../output/amazon/scenario3/group1_accuracy_var_map.png')
plt.close()
plt.clf()
##group2 mean
plt.figure()
sns.heatmap(
    (accuracy_mean[1,:,:]-no_obs_accuracy_mean[1])/1000,
    cmap='bwr',
    vmin=-0.5,
    vmax=0.5,
    yticklabels=[0.025,0.05,0.1,0.2],
    xticklabels=[11,21,41,81,101,201])

plt.savefig('../../output/amoc/scenario/group_accuracy_mean_map.png')
plt.close()
plt.clf()
##group2 var
plt.figure()
sns.heatmap(
    accuracy_var[1,:,:]/1000000,
    cmap='gist_gray',
    yticklabels=[0.025,0.05,0.1,0.2],
    xticklabels=[11,21,41,81,101,201])

plt.savefig('../../output/amoc/scenario/group2_accuracy_var_map.png')
plt.close()
plt.clf()
##group3 mean
plt.figure()
sns.heatmap(
    ((-1)*accuracy_mean[2,:,:]+no_obs_accuracy_mean[2])/1000,
    cmap='bwr',
    vmin=-0.5,
    vmax=0.5,
    yticklabels=[0.025,0.05,0.1,0.2],
    xticklabels=[11,21,41,81,101,201])

plt.savefig('../../output/amoc/scenario/group3_accuracy_mean_map.png')
plt.close()
plt.clf()
##group3 var
plt.figure()
sns.heatmap(
    accuracy_var[2,:,:]/1000000,
    cmap='gist_gray',
    yticklabels=[0.025,0.05,0.1,0.2],
    xticklabels=[11,21,41,81,101,201])

plt.savefig('../../output/amoc/scenario/group3_accuracy_var_map.png')
plt.close()
plt.clf()