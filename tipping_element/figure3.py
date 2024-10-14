##import libralies
import pandas as pd
import matplotlib.pyplot as plt
import os 
import yaml
##internal variability and initial accuracy
name = os.environ['NAME']
datadir=os.environ['EXDATADIR']
outdir=os.environ['EXOUTDIR']
ex_num=os.environ['EXNUM']

no_obs_data_path0= datadir + '/experiment1/tip_num_non_obs_amp0.csv'
no_obs_data_path1= datadir + '/experiment1/tip_num_non_obs_amp1.csv'
no_obs_data_path2= datadir + '/experiment1/tip_num_non_obs_amp2.csv'
no_obs_data_path3= datadir + '/experiment1/tip_num_non_obs_amp3.csv'
no_obs_data_path4= datadir + '/experiment1/tip_num_non_obs_amp4.csv'

obs_data_path0= datadir + '/experiment2/tip_num_amp0.csv'
obs_data_path1= datadir + '/experiment2/tip_num_amp1.csv'
obs_data_path2= datadir + '/experiment2/tip_num_amp2.csv'
obs_data_path3= datadir + '/experiment2/tip_num_amp3.csv'
obs_data_path4= datadir + '/experiment2/tip_num_amp4.csv'

obs200_data_path0= datadir + '/experiment1/tip_num_amp0.csv'
obs200_data_path1= datadir + '/experiment1/tip_num_amp1.csv'
obs200_data_path2= datadir + '/experiment1/tip_num_amp2.csv'
obs200_data_path3= datadir + '/experiment1/tip_num_amp3.csv'
obs200_data_path4= datadir + '/experiment1/tip_num_amp4.csv'


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

data200_0=pd.read_csv(obs200_data_path0,header=None)
data200_1=pd.read_csv(obs200_data_path1,header=None)
data200_2=pd.read_csv(obs200_data_path2,header=None)
data200_3=pd.read_csv(obs200_data_path3,header=None)
data200_4=pd.read_csv(obs200_data_path4,header=None)

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

g1data200_0=data200_0[data200_0.iloc[:,0]==1]
g1data200_1=data200_1[data200_1.iloc[:,0]==1]
g1data200_2=data200_2[data200_2.iloc[:,0]==1]
g1data200_3=data200_3[data200_3.iloc[:,0]==1]
g1data200_4=data200_4[data200_4.iloc[:,0]==1]

g1data0=pd.concat([g1data0,g1data200_0],ignore_index=True)
g1data1=pd.concat([g1data1,g1data200_1],ignore_index=True)
g1data2=pd.concat([g1data2,g1data200_2],ignore_index=True)
g1data3=pd.concat([g1data3,g1data200_3],ignore_index=True)
g1data4=pd.concat([g1data4,g1data200_4],ignore_index=True)

if name == 'AMOC':
    with open(ex_num+'_' +name +'.yml') as file:
        config = yaml.safe_load(file.read())
else :
    with open(ex_num +'.yml') as file:
        config = yaml.safe_load(file.read())

accuracy_list = config['accuracy']
obs_fs_list = config['obs_fs']

num_list=[list[0] for list in obs_fs_list]
if name == 'AMOC':
    num_list.append(100)
else:
    num_list.append(40)

print(num_list)
colorlist = ["r", "g", "b", "c", "m", "y"]
err_list=accuracy_list

plt.rcParams["font.size"] = 25

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


        ax.plot([0,1,2,3,4],[acc0,acc1,acc2,acc3,acc4],"-",label='Err={}%'.format(err*100),marker='o',markersize=8,color=colorlist[j])
    plt.legend(fontsize=14)
    if name =='AMOC':
        plt.title('Amplitude-accuracy lead time={} years'.format((100-num)*2+2),fontsize=30)
    else:
        plt.title('Amplitude-accuracy lead time={} years'.format((40-num)*5+2),fontsize=30)
    fig.savefig(outdir + '/experiment2/amplitude_leadtime_{}.png'.format((40-num)*5+2))
    fig.clf()
    plt.close()
plt.rcParams["font.size"] = 25
for j,err in enumerate(err_list):
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(-0.1,4.1)
    ax.set_ylim(-0.46,0.46)
    ax.set_xlabel('forcing variability amplitude')
    ax.set_ylabel('$\Delta$ Accuracy')
    for i,num in enumerate(num_list):
        acc0=(g1data0[(g1data0.iloc[:,1]==err) & (g1data0.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata0.iloc[:,3].mean())/1000
        acc1=(g1data1[(g1data1.iloc[:,1]==err) & (g1data1.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata1.iloc[:,3].mean())/1000
        acc2=(g1data2[(g1data2.iloc[:,1]==err) & (g1data2.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata2.iloc[:,3].mean())/1000
        acc3=(g1data3[(g1data3.iloc[:,1]==err) & (g1data3.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata3.iloc[:,3].mean())/1000
        acc4=(g1data4[(g1data4.iloc[:,1]==err) & (g1data4.iloc[:,2]==num)].iloc[:,3].mean()-g1ndata4.iloc[:,3].mean())/1000

        if name =='AMOC':
            ax.plot([0,1,2,3,4],[acc0,acc1,acc2,acc3,acc4],"-",label='lead time={} years'.format((100-num)*2+2),marker='o',markersize=8,color=colorlist[i])
        else:
            ax.plot([0,1,2,3,4],[acc0,acc1,acc2,acc3,acc4],"-",label='lead time={} years'.format((40-num)*5+2),marker='o',markersize=8,color=colorlist[i])
    plt.legend(fontsize=14)
    plt.title('Amplitude-accuracy Err={}%'.format(err*100),fontsize=30)
    fig.savefig(outdir + '/experiment2/figure3_err_{}.png'.format(err*100))
    fig.clf()
    plt.close()

plt.rcParams["font.size"] = 25
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


    ax.plot([0,1,2,3,4],[acc0,acc1,acc2,acc3,acc4],"-",label='Err={}%'.format(err*100),marker='o',markersize=8,color=colorlist[j])
plt.legend(fontsize=14)
plt.title('Amplitude-accuracy lead time mean',fontsize=30)
fig.savefig(outdir + '/experiment2/figure3_mean.png')
fig.clf()
plt.close()
