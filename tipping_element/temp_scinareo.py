import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
##function setting
def T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=0):
    ##This is simpler version of temperature profile
    ##Set dummy threshold d1,d2,d3
    d1=(Tth+dTex-Tst)/r
    d2=d1+dtex-dTex/s-dTex/r
    d3=d2+(Tth+dTex-Te)/s

    t=np.array([i*dt for i in range(steps)])
    T=(Tst+r*t)*(t<d1) \
    +(Tth+dTex)*(t>=d1)*(t<d2) \
    +(Tth+dTex-s*(t-d2))*(t<d3)*(t>=d2) \
    +Te*(t>=d3)

    T+=amp*np.random.randn(steps)*dt**(0.5)
    print('temperature noise :'+str(amp*dt**(0.5)))

    return T

def amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf,v):
    g=g0*(1-(((Tf+alpha*(1-v))-Topt)/beta)**2)
    k=g*v*(1-v)-gamma*v

    return k

def Runge_Kutta_dynamics(v,Tl,g,steps,Tf,dt=0.1,epsilon=1):
    ##set hyperparams
    alpha=5
    beta=10
    Topt=28
    g0=2
    gamma=0.2

    v[0]=0.8
    Tl[0]=Tf[0]+(1-v[0])*alpha
    g[0]=g0*(1-((Tl[0]-Topt/beta))**2)

    for step in range(steps-1):
        k1=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v[step])
        k2=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v[step]+dt/2*k1)
        k3=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v[step]+dt/2*k2)
        k4=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v[step]+dt*k3)

        #print('k1={},k2={},k3={},k4={}'.format(k1,k2,k3,k4))

        v[step+1]=v[step]+dt/6*(k1+2*k2+2*k3+k4)/epsilon

        Tl[step+1]=Tf[step+1]+alpha*(1-v[step+1])
        g[step+1]=g0*(1-((Tl[step+1]-Topt)/beta)**2)
##parameter setting

Tth=34.7
Te=34.0
dtex=80
dt=0.1
steps=10000

params=[
    [32.9,(Tth-32.9)/202,0.008,-0.1,-10,'r']
    #[32.9,(Tth-32.9)/202,0.008,0.2,100,'g'],
    #[32.9,(Tth-32.9)/202,0.008,0.01,40,'b']
]
##温度のプロファイルを図示する
T=np.array([t*dt for t in range(steps)])
output='../../output/amazon'
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax1.set_xlim(0,dt*steps)
ax1.set_ylim(30,40)
ax1.set_ylabel('Tf')
ax1.hlines(Tth,0,1000,linestyles='dotted')

ax2=fig.add_subplot(2,1,2)
ax2.set_xlim(0,dt*steps)
ax2.set_ylim(-0.1,1.1)
ax2.set_ylabel('v')
ax2.set_xlabel('time(years)')

for T_start,r,s,dTex,dtex,c in params:
    Tl_obs=np.zeros((steps,))
    v_obs=np.zeros((steps,))
    g_obs=np.zeros((steps,))
    epsilon=1
    np.random.seed(1)
    Tf=T_develop2(dt,T_start,Tth,dTex,Te,dtex,r,s,steps,amp=1)
    Runge_Kutta_dynamics(v_obs,Tl_obs,g_obs,steps,Tf,epsilon=epsilon)
    ax1.plot(T,Tf,color=c)
    ax2.plot(T,v_obs,color=c)
ax1.set_title('temperature & v profile')
fig.savefig(output+'/scinareo1_amp4.png')
fig.clf()
