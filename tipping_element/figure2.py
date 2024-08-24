##import libralies
import pandas as pd
import matplotlib.pyplot as plt
import os
##internal variability and initial accuracy
datadir=os.environ['EXDATADIR']
outdir=os.environ['EXOUTDIR']

no_obs_data_path0=datadir + '/experiment1/tip_num_non_obs_amp0.csv'
no_obs_data_path1=datadir + '/experiment1/tip_num_non_obs_amp1.csv'
no_obs_data_path2=datadir + '/experiment1/tip_num_non_obs_amp2.csv'
no_obs_data_path3=datadir + '/experiment1/tip_num_non_obs_amp3.csv'
no_obs_data_path4=datadir + '/experiment1/tip_num_non_obs_amp4.csv'

obs_data_path0=datadir + '/experiment1/tip_num_amp0.csv'
obs_data_path1=datadir + '/experiment1/tip_num_amp1.csv'
obs_data_path2=datadir + '/experiment1/tip_num_amp2.csv'
obs_data_path3=datadir + '/experiment1/tip_num_amp3.csv'
obs_data_path4=datadir + '/experiment1/tip_num_amp4.csv'

ndata0=pd.read_csv(no_obs_data_path0,header=None)
ndata1=pd.read_csv(no_obs_data_path1,header=None)
ndata2=pd.read_csv(no_obs_data_path2,header=None)
ndata3=pd.read_csv(no_obs_data_path3,header=None)
ndata4=pd.read_csv(no_obs_data_path4,header=None)

data0=pd.read_csv(obs_data_path0,header=None)
data1=pd.read_csv(obs_data_path1,header=None)
data2=pd.read_csv(obs_data_path2,header=None)
data3=pd.read_csv(obs_data_path3,header=None)
data4=pd.read_csv(obs_data_path4,header=None)
##
g1ndata0=ndata0[ndata0.iloc[:,0]==1]
g1ndata1=ndata1[ndata1.iloc[:,0]==1]
g1ndata2=ndata2[ndata2.iloc[:,0]==1]
g1ndata3=ndata3[ndata3.iloc[:,0]==1]
g1ndata4=ndata4[ndata4.iloc[:,0]==1]

g1data0=data0[data0.iloc[:,0]==1]
g1data1=data1[data1.iloc[:,0]==1]
g1data2=data2[data2.iloc[:,0]==1]
g1data3=data3[data3.iloc[:,0]==1]
g1data4=data4[data4.iloc[:,0]==1]

num_list=[10,20,40,80,100,200]
colorlist = ["r", "g", "b", "c", "m", "y"]
err_list=[0.01, 0.025, 0.05, 0.1, 0.2]

plt.rcParams["font.size"] = 15

for i,num in enumerate(num_list):
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(-0.1,4.1)
    ax.set_ylim(-0.46,0.46)
    ax.set_xlabel('forcing variability amplitude')
    ax.set_ylabel('$\Delta$ Accuracy')
    for j,err in enumerate(err_list):
        acc0=(g1data0[(g1data0.iloc[:,1]==err) & (g1data0.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata0.iloc[:,3].mean())/1000
        acc1=(g1data1[(g1data1.iloc[:,1]==err) & (g1data1.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata1.iloc[:,3].mean())/1000
        acc2=(g1data2[(g1data2.iloc[:,1]==err) & (g1data2.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata2.iloc[:,3].mean())/1000
        acc3=(g1data3[(g1data3.iloc[:,1]==err) & (g1data3.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata3.iloc[:,3].mean())/1000
        acc4=(g1data4[(g1data4.iloc[:,1]==err) & (g1data4.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata4.iloc[:,3].mean())/1000


        ax.plot([0,1,2,3,4],[acc0,acc1,acc2,acc3,acc4],"-",label='Err={}'.format(err*10),marker='o',markersize=8,color=colorlist[j])
    plt.legend()
    fig.savefig(outdir+ '/experiment1/amplitude_num_{}_1.png'.format(num))
    fig.clf()
    plt.close()

##
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(1,1,1)
ax.set_xlim(-0.1,4.1)
ax.set_ylim(-0.46,0.46)
ax.set_xlabel('forcing variability amplitude')
ax.set_ylabel('$\Delta$ Accuracy')
for j,err in enumerate(err_list):
    acc0=(g1data0[g1data0.iloc[:,1]==err].iloc[:,3].mean()-g1ndata0.iloc[:,3].mean())/1000
    acc1=(g1data1[g1data1.iloc[:,1]==err].iloc[:,3].mean()-g1ndata1.iloc[:,3].mean())/1000
    acc2=(g1data2[g1data2.iloc[:,1]==err].iloc[:,3].mean()-g1ndata2.iloc[:,3].mean())/1000
    acc3=(g1data3[g1data3.iloc[:,1]==err].iloc[:,3].mean()-g1ndata3.iloc[:,3].mean())/1000
    acc4=(g1data4[g1data4.iloc[:,1]==err].iloc[:,3].mean()-g1ndata4.iloc[:,3].mean())/1000


    ax.plot([0,1,2,3,4],[acc0,acc1,acc2,acc3,acc4],"-",label='Err={}'.format(err*10),marker='o',markersize=8,color=colorlist[j])
plt.legend()
fig.savefig(outdir + '/experiment1/amplitude_num_1.png')
fig.clf()
plt.close()
##
g2ndata0=ndata0[ndata0.iloc[:,0]==2]
g2ndata1=ndata1[ndata1.iloc[:,0]==2]
g2ndata2=ndata2[ndata2.iloc[:,0]==2]
g2ndata3=ndata3[ndata3.iloc[:,0]==2]
g2ndata4=ndata4[ndata4.iloc[:,0]==2]

g2data0=data0[data0.iloc[:,0]==2]
g2data1=data1[data1.iloc[:,0]==2]
g2data2=data2[data2.iloc[:,0]==2]
g2data3=data3[data3.iloc[:,0]==2]
g2data4=data4[data4.iloc[:,0]==2]

num_list=[10,20,40,80,100,200]
colorlist = ["r", "g", "b", "c", "m", "y"]
err_list=[0.01, 0.025, 0.05, 0.1, 0.2]

for i,num in enumerate(num_list):
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(-0.1,4.1)
    ax.set_ylim(-0.70,0.70)
    ax.set_xlabel('forcing variability amplitude')
    ax.set_ylabel('$\Delta$ Accuracy')
    for j,err in enumerate(err_list):
        acc0=(g2data0[(g2data0.iloc[:,1]==err) & (g2data0.iloc[:,2]==num)].iloc[:,3].mean()-g2ndata0.iloc[:,3].mean())/1000
        acc1=(g2data1[(g2data1.iloc[:,1]==err) & (g2data1.iloc[:,2]==num)].iloc[:,3].mean()-g2ndata1.iloc[:,3].mean())/1000
        acc2=(g2data2[(g2data2.iloc[:,1]==err) & (g2data2.iloc[:,2]==num)].iloc[:,3].mean()-g2ndata2.iloc[:,3].mean())/1000
        acc3=(g2data3[(g2data3.iloc[:,1]==err) & (g2data3.iloc[:,2]==num)].iloc[:,3].mean()-g2ndata3.iloc[:,3].mean())/1000
        acc4=(g2data4[(g2data4.iloc[:,1]==err) & (g2data4.iloc[:,2]==num)].iloc[:,3].mean()-g2ndata4.iloc[:,3].mean())/1000


        ax.plot([0,1,2,3,4],[-acc0,-acc1,-acc2,-acc3,-acc4],"-",label='Err={}'.format(err*10),marker='o',markersize=8,color=colorlist[j])
    plt.legend()
    fig.savefig(outdir+ '/experiment1/amplitude_num_{}_2.png'.format(num))
    fig.clf()
    plt.close()

##
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(1,1,1)
ax.set_xlim(-0.1,4.1)
ax.set_ylim(-0.70,0.70)
ax.set_xlabel('forcing variability amplitude')
ax.set_ylabel('$\Delta$ Accuracy')
for j,err in enumerate(err_list):
    acc0=(g2data0[g2data0.iloc[:,1]==err].iloc[:,3].mean()-g2ndata0.iloc[:,3].mean())/1000
    acc1=(g2data1[g2data1.iloc[:,1]==err].iloc[:,3].mean()-g2ndata1.iloc[:,3].mean())/1000
    acc2=(g2data2[g2data2.iloc[:,1]==err].iloc[:,3].mean()-g2ndata2.iloc[:,3].mean())/1000
    acc3=(g2data3[g2data3.iloc[:,1]==err].iloc[:,3].mean()-g2ndata3.iloc[:,3].mean())/1000
    acc4=(g2data4[g2data4.iloc[:,1]==err].iloc[:,3].mean()-g2ndata4.iloc[:,3].mean())/1000


    ax.plot([0,1,2,3,4],[-acc0,-acc1,-acc2,-acc3,-acc4],"-",label='Err={}'.format(err*10),marker='o',markersize=8,color=colorlist[j])
plt.legend()
fig.savefig(outdir + '/experiment1/amplitude_num_2.png')
fig.clf()
plt.close()