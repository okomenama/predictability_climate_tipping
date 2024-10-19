import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

def find_mode_of_intervals(data, bin_width, min_val, max_val):
    """
    連続データを一定間隔で区切り、最頻値の階級を調べる関数

    Parameters:
    data (array): 連続値が格納された配列
    bin_width (float): 階級の幅

    Returns:
    tuple: 最頻値となるビンの範囲とその頻度
    """
    
    # ビンの境界値を作成
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 各値がどのビンに属するかを確認し、ヒストグラムを作成
    histogram, bin_edges = np.histogram(data, bins=bins)
    
    # 最頻値のビンを見つける（ヒストグラム内の最大頻度を持つビン）
    mode_index = np.argmax(histogram)
    
    # 最頻値となるビンの範囲
    mode_bin = bin_edges[mode_index]+bin_width/2
    
    return mode_bin, histogram[mode_index]

plt.rcParams["font.size"] = 25

acc_mean_mean_list=[]
acc_mean_mode_list = []
pre_mean_mean_list=[]

acc_std_list=[]
pre_std_list=[]
for amp in range(5):
    data='../data/rev_amazon/experiment_acc/tip_num_amp'+str(amp)+'_spread.csv'

    df=pd.read_csv(data,header=None)
    #print(df)

    acc_mean_list=[]
    pre_mean_list=[]
    sacc_std_list=[]
    spre_std_list=[]
    var_list=[]

    sacc_mean_mean_list=[]
    sacc_mean_mode_list = []
    spre_mean_mean_list=[]

    for i in range(2800):
        acc_mean=df.iloc[i*1000:(i+1)*1000,3].mean()
        pre_mean=df.iloc[i*1000:(i+1)*1000,4].mean()

        acc_mean_list.append(acc_mean)
        pre_mean_list.append(pre_mean)

    for j in range(28):
        acc_std=np.array(acc_mean_list)[j*100:(j+1)*100].std()
        pre_std=np.array(pre_mean_list)[j*100:(j+1)*100].std()

        acc_mean_mean= np.array(acc_mean_list)[j*100:(j+1)*100].mean()
        acc_mean_mode = find_mode_of_intervals(np.abs(np.array(acc_mean_list)[j*100:(j+1)*100]), 0.1, 0, 1)
        pre_mean_mean=np.array(pre_mean_list)[j*100:(j+1)*100].mean()

        sacc_mean_mean_list.append(acc_mean_mean)
        sacc_mean_mode_list.append(acc_mean_mode[0])
        spre_mean_mean_list.append(pre_mean_mean)

        sacc_std_list.append(acc_std)
        spre_std_list.append(pre_std)

    acc_mean_mean_list.append(sacc_mean_mean_list)
    acc_mean_mode_list.append(sacc_mean_mode_list)
    pre_mean_mean_list.append(spre_mean_mean_list)
    acc_std_list.append(sacc_std_list)
    pre_std_list.append(spre_std_list)

print(acc_mean_mean_list)
print(pre_mean_mean_list)

fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(1,1,1)
clist=[(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 0.8), (0.0, 0.0, 1.0, 0.6), (0.0, 0.0, 1.0, 0.4),
       (0.3, 0.0, 0.7, 1.0), (0.3, 0.0, 0.7, 0.8), (0.3, 0.0, 0.7, 0.6), (0.3, 0.0, 0.7, 0.4),
        (0.5, 0.0, 0.5, 1.0),(0.5, 1.0, 0.5, 0.8),(0.5, 1.0, 0.5,0.6), (0.5, 1.0, 0.5, 0.4),
        (0.7, 0.0, 0.3, 1.0), (0.7, 0.0, 0.3, 0.8), (0.7, 0.0, 0.3, 0.6), (0.7, 0.0, 0.3, 0.4),
        (1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.8), (1.0, 0.0, 0.0, 0.6), (1.0, 0.0, 0.0, 0.4)]
llist=['0.01','0.025','0.05','0.1']
print('ACC 分布')
print(acc_mean_mode_list)
l = 6
markers = ["o", ",", "x", "v"]

for j in range(5):
    #ax.errorbar(acc_mean_mean_list[j][slice(6,None,7)],pre_mean_mean_list[j][slice(6,None,7)],xerr=acc_std_list[j][slice(6,None,7)],yerr=pre_std_list[j][slice(6,None,7)],capsize=5, fmt='o', markersize=5, ecolor=clist[4*j], markeredgecolor = clist[4*j], color='w',label='$\sigma_{}={}$'.format('{sig}',j))
    for k in range(4):
        if k == 1:
            ax.scatter(acc_mean_mean_list[j][6+7*k],pre_mean_mean_list[j][6+7*k], marker=markers[k], color = clist[4*j],label='$\sigma_{}={}$'.format('{sig}',j), s=500)
        else :
            ax.scatter(acc_mean_mean_list[j][6+7*k],pre_mean_mean_list[j][6+7*k], marker=markers[k], color = clist[4*j], s=500)

ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("ACC")
ax.set_ylabel("Accuracy")
#ax.set_title('The distribution of Correlation and Accuracy $\sigma_{}={}$'.format('{sig}',amp))
ax.set_title('The distribution of ACC and Accuracy')
ax.legend()
fig.savefig('../output/rev_amazon/corr_accuracy_test.png')
fig.clf()
plt.close()
'''
    df=df.rename(columns={0:'scenario',1:'error',2:'obs_num',3:'corr',4:'tip'})
    fig, ax = plt.subplots(figsize=(30, 10))
    sns.boxplot(x = "error", y = "corr", data = df, hue = "obs_num",ax=ax)
    fig.savefig('../output/rev_amazon/correlation.png')

    group_data = df.groupby(['error','obs_num'])
    for i, group in group_data:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(-1,1)
        ax.hist(group.iloc[:,3], bins = 20, range = (-1,1))
        err = group.iat[0,1]
        num = group.iat[0,2]
        plt.title('ERR={}, NUM={}'.format(err,num))
        fig.savefig('../output/rev_amazon/correlation'+str(amp)+'_hist_{}_{}.'.format(err,num))
        fig.clf()
        plt.close()
'''
acc_mean_mean_list=[]
acc_mean_mode_list = []
pre_mean_mean_list=[]

acc_std_list=[]
pre_std_list=[]
for amp in range(5):
    data='../data/rev_AMOC/experiment_acc/tip_num_amp'+str(amp)+'acc.csv'

    df=pd.read_csv(data,header=None)
    #print(df)

    acc_mean_list=[]
    pre_mean_list=[]
    sacc_std_list=[]
    spre_std_list=[]
    var_list=[]

    sacc_mean_mean_list=[]
    sacc_mean_mode_list = []
    spre_mean_mean_list=[]

    for i in range(400):
        acc_mean=df.iloc[i*1000:(i+1)*1000,3].mean()
        pre_mean=df.iloc[i*1000:(i+1)*1000,4].mean()

        acc_mean_list.append(acc_mean)
        pre_mean_list.append(pre_mean)

    for j in range(4):
        acc_std=np.array(acc_mean_list)[j*100:(j+1)*100].std()
        pre_std=np.array(pre_mean_list)[j*100:(j+1)*100].std()

        acc_mean_mean= np.array(acc_mean_list)[j*100:(j+1)*100].mean()
        acc_mean_mode = find_mode_of_intervals(np.abs(np.array(acc_mean_list)[j*100:(j+1)*100]), 0.1, 0, 1)
        pre_mean_mean=np.array(pre_mean_list)[j*100:(j+1)*100].mean()

        sacc_mean_mean_list.append(acc_mean_mean)
        sacc_mean_mode_list.append(acc_mean_mode[0])
        spre_mean_mean_list.append(pre_mean_mean)

        sacc_std_list.append(acc_std)
        spre_std_list.append(pre_std)

    acc_mean_mean_list.append(sacc_mean_mean_list)
    acc_mean_mode_list.append(sacc_mean_mode_list)
    pre_mean_mean_list.append(spre_mean_mean_list)
    acc_std_list.append(sacc_std_list)
    pre_std_list.append(spre_std_list)

print(len(acc_mean_mean_list))

fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(1,1,1)
llist=['0.025','0.05','0.1','0.2']
print('ACC 分布')
print(acc_mean_mode_list)
l = 0
for j in range(5):
    for k in range(4):
        if k==1:
            ax.scatter(acc_mean_mean_list[j][k],pre_mean_mean_list[j][k], color = clist[4*j], label = '$\sigma_{}={}$'.format('{sig}', j), marker=markers[k], s=500)
        else :
            ax.scatter(acc_mean_mean_list[j][k],pre_mean_mean_list[j][k], color =clist[4*j], marker=markers[k], s = 500)
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
ax.set_xlabel("ACC")
ax.set_ylabel("Accuracy")
#ax.set_title('The distribution of Correlation and Accuracy $\sigma_{}={}$'.format('{sig}',amp))
ax.set_title('The distribution of ACC and Accuracy')
ax.legend()
fig.savefig('../output/rev_AMOC/corr_accuracy_test.png')
fig.clf()
plt.close()