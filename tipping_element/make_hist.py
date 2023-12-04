import pandas as pd
import matplotlib.pyplot as plt
data_path='../data/amazon/tip_num3.csv'
no_obs_data_path='../data/amazon/tip_num_non_obs3.csv'

data=pd.read_csv(data_path,header=None)
no_obs_data=pd.read_csv(no_obs_data_path,header=None)

group_data=data.groupby([0,1,2])

for i,group in group_data:
    print(group)
    group.iloc[:,2]=2000/(group.iloc[:,2]-1)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('number of tipping particle')
    ax.hist(group.iloc[:,3],range=(0,1000),color='b')
    ax.hist(no_obs_data[no_obs_data[0]==group.iat[0,0]].iloc[:,3],range=(0,1000),color=(1, 0, 0, 0.2))
    plt.title(
        'group:{},v_obs:{},fs:{}'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.savefig(
        '../../output/amazon/scenario3/group3_{}_{}_{}.png'.format(
            group.iat[0,0],group.iat[0,1],group.iat[0,2]))
    plt.clf()