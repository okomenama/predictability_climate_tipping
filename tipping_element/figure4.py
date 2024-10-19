import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 25

result=[]
for amp in [1,2,3,4]:
    data_freq='../data/rev_amazon/experiment2/tip_num_amp{}.csv'.format(amp)
    #data_sn='../data/final_result/amazon/signal_ratio.csv'
    data_sig='../output/amazon/rms/np_temp_amp{}.csv'.format(amp)

    df_freq=pd.read_csv(data_freq,header=None)
    #df_sn=pd.read_csv(data_sn,header=None)
    df_sig=pd.read_csv(data_sig,header=None)

    #arr_sn=df_sn.iloc[0,:].values
    arr_sig=df_sig.iloc[:,0].values
    print(arr_sig)
    df_freq=df_freq[df_freq.iloc[:,0]==1]

    ##観測の数が100だと1,90だと201,80だと401,70だと601を返すような関数を考える
    group_data=df_freq.groupby([1,2])

    print(len(group_data))
    for i,group in group_data:
        group_obs_num=group.iat[0,2]
        group_noise=group.iat[0,1]

        #group_sn=amp/group_noise/abs(arr_sn[0:(-20)*(100-group_obs_num)-1].mean())*0.1**0.5
        group_sn=arr_sig[0:(-50)*(40-group_obs_num)-1].max()/group_noise
        group_ave=group.iloc[:,3].mean()/1000
        group_var=group.iloc[:,3].var()/1000000
        if group_noise != 0.1:
            result.append([group_sn,group_ave,group_var])
        print(group_noise)

plt.figure(figsize=(8,7))
result=np.array(result)
print(len(result))
plt.ylim(0.1,1.05)
plt.errorbar(result[:,0],result[:,1],np.sqrt(result[:,2]),capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
plt.title('TRIFFID S/N-Tipping Accuracy')
plt.xlabel('S/N')
plt.ylabel('Accuracy')
plt.savefig('../output/rev_amazon/sn_prob_mean_m.png')
plt.clf()


plt.figure(figsize=(8,7))

result=[]
for amp in [1,2,3,4]:
    data_freq='../data/rev_amoc/experiment2/tip_num_amp{}.csv'.format(amp)
    #data_sn='../data/final_result/amoc/signal_ratio.csv'
    data_sig='../output/amoc/rms/np_temp_amp{}.csv'.format(amp)

    df_freq=pd.read_csv(data_freq,header=None)
    #df_sn=pd.read_csv(data_sn,header=None)
    df_sig=pd.read_csv(data_sig,header=None)

    #arr_sn=df_sn.iloc[0,:].values
    arr_sig=df_sig.iloc[:,0].values
    #print(arr_sn)

    df_freq=df_freq[df_freq.iloc[:,0]==1]

    ##観測の数が100だと1,90だと201,80だと401,70だと601を返すような関数を考える
    group_data=df_freq.groupby([1,2])

    for i,group in group_data:
        group_obs_num=group.iat[0,2]
        group_noise=group.iat[0,1]*10
        group_noise_num = 0*(group_noise==0.1) + 1*(group_noise==0.25) + 2*(group_noise==0.5) + 3*(group_noise==1) + 4*(group_noise==0.2)

        #group_sn=amp/group_noise/abs(arr_sn[2000:(-20)*(100-group_obs_num)-1].mean())*0.1**0.5
        group_sn=arr_sig[0:(-50)*(40-group_obs_num)-1].max()/group_noise
        group_ave=group.iloc[:,3].mean()/1000
        group_var=group.iloc[:,3].var()/1000000
        result.append([group_sn,group_ave,group_var])

plt.title('AMOC S/N-Tipping Accuracy')
plt.xlabel('S/N')
plt.ylabel('Accuracy')
plt.savefig('../output/rev_amoc/sn_prob_mean_m.png')

plt.figure(figsize=(8,7))
result=np.array(result)
plt.ylim(0.3,1.05)
plt.errorbar(result[:,0],result[:,1],np.sqrt(result[:,2]),capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
plt.title('AMOC S/N-Tipping Accuracy')
plt.xlabel('S/N')
plt.ylabel('Accuracy')
plt.savefig('../output/rev_amoc/sn_prob_mean_m.png')
plt.clf()   

