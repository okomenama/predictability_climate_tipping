import amazon
import numpy as np
import matplotlib.pyplot as plt

def PCA(X,n):
    ##行列形式でデータをinputする
    print(len(X))
    X=np.array(X)
    X_mean=np.mean(X,axis=0)
    print(X.shape)
    X_cov=np.cov((X-X_mean).T)
    print(X_cov.shape)
    u,s,v=np.linalg.svd(X_cov)
    
    d=s[:n]
    u_d=u[:,:n]

    pn_la=s[:n]
    pn_vec=np.dot(u_d,np.diag(d))
    print(pn_vec.shape)
    ratio=np.sum(d**2)/np.sum(s**2)
    return pn_la,pn_vec.T,ratio
##param setting
window=50
steps=4096
dt=0.1##サンプリングレートは10
iters=100

fft_as_list=[]

for T0 in range(25,35,1):
    v=np.zeros((steps,iters))
    Tf=np.array([T0 for i in range(steps)])
    Tl=np.zeros((steps,iters))
    g=np.zeros((steps,iters))

    amazon.Runge_Kutta_dynamics(v,Tl,g,steps,Tf,iters,dt,noise=1,amp=0.05)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(v[500:500+2048,0],v[501:501+2048,0])
    ax.set_xlim(0.6,1)
    ax.set_ylim(0.6,1)
    fig.savefig('../../output/amazon/spectrum/vi_vi1'+str(T0)+'.png')
    fig.clf()

    ratios=[]
    for i in range(iters):
        pn_la,pn_vec,ratio=PCA(
            np.array(
                [v[500:500+2048,i]
                ,v[501:501+2048,i]]).T
                ,1)
        print(str(1)+'番目までの寄与率'+str(ratio))
        ratios.append(ratio)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(ratios,bins=20,range=(0.98,1.0))
    fig.savefig('../../output/amazon/spectrum/cov_v_'+str(T0)+'.png')
    fig.clf()

'''
for i in range(iters):
    f_v = np.fft.fft(v[500:500+2048,i])
    fft_as = np.abs(f_v)**2
    fft_as_list.append(fft_as[1:100])

# 周波数成分の計算
fft_freq = np.fft.fftfreq(len(f_v),d=dt)
print(len(fft_freq[fft_freq>0]))
fft_period=1/fft_freq[1:100]  
# プロット
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(fft_freq[1:100], fft_as[1:100])
ax.set_xlabel('frequency')
fig.savefig('../../output/amazon/spectrum/pow_spectrum.png')
fig.clf()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(fft_period, fft_as[1:100])
ax.set_xlabel('period')

fig.savefig('../../output/amazon/spectrum/pow_spectrum_period.png')
fig.clf()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot([500+i*dt for i in range(steps-500)],v[500:,1])
fig.savefig('../../output/amazon/spectrum/v.png')

n=5
pn_la,pn_vec,ratio=PCA(fft_as_list,n)
print('最大固有値:'+str(pn_la[0]))
print('最大固有ベクトル'+str(pn_vec[0,:]))
print(str(n)+'番目までの寄与率'+str(ratio))

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(pn_vec[0,:])
'''