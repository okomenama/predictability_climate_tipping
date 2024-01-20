import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot()

for amp in [1,2,3,4]:
    data_freq='../data/final_result/amazon/tip_num_freq_amp{}.csv'.format(amp)
    data_sn='../data/final_result/amazon/signal_ratio.csv'

    df_freq=pd.read_csv(data_freq,header=None)
    df_sn=pd.read_csv(data_sn,header=None)

    arr_sn=df_sn.iloc[0,:].values
    print(arr_sn)

    df_freq=df_freq[df_freq.iloc[:,0]==1]

    ##観測の数が100だと1,90だと201,80だと401,70だと601を返すような関数を考える
    group_data=df_freq.groupby([1,2])

    result=[]
    for i,group in group_data:
        group_obs_num=group.iat[0,2]
        group_noise=group.iat[0,1]

        group_sn=amp/group_noise/abs(arr_sn[0:(-20)*(100-group_obs_num)-1].mean())
        group_ave=group.iloc[:,3].mean()/1000
        group_var=group.iloc[:,3].var()/1000000
        result.append([group_sn,group_ave,group_var])

plt.figure()
result=np.array(result)
plt.ylim(0.5,1.05)
plt.errorbar(result[:,0],result[:,1],np.sqrt(result[:,2]),capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
plt.title('TRIFFID S/N-Tipping probability')
plt.xlabel('S/N')
plt.ylabel('Probability')
plt.savefig('../../output/final_result/amazon/sn_prob_mean.png')
plt.clf()


plt.figure()

for amp in [1,2,3,4]:
    data_freq='../data/final_result/amoc/tip_num_freq_amp{}.csv'.format(amp)
    data_sn='../data/final_result/amoc/signal_ratio.csv'

    df_freq=pd.read_csv(data_freq,header=None)
    df_sn=pd.read_csv(data_sn,header=None)

    arr_sn=df_sn.iloc[0,:].values
    print(arr_sn)

    df_freq=df_freq[df_freq.iloc[:,0]==1]

    ##観測の数が100だと1,90だと201,80だと401,70だと601を返すような関数を考える
    group_data=df_freq.groupby([1,2])

    result=[]
    for i,group in group_data:
        group_obs_num=group.iat[0,2]
        group_noise=group.iat[0,1]*10

        group_sn=amp/group_noise/abs(arr_sn[2000:(-20)*(100-group_obs_num)-1].mean())
        group_ave=group.iloc[:,3].mean()/1000
        group_var=group.iloc[:,3].var()/1000000
        result.append([group_sn,group_ave,group_var])

result=np.array(result)
plt.ylim(0.5,1.05)
plt.errorbar(result[:,0],result[:,1],np.sqrt(result[:,2]),capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
plt.title('AMOC S/N-Tipping probability')
plt.xlabel('S/N')
plt.ylabel('Probability')
plt.savefig('../../output/final_result/amoc/sn_prob_mean.png')
plt.clf()   

