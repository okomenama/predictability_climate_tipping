import numpy as np
import matplotlib.pyplot as plt
import amazon
import AMOC

##calculate signal noise ratio
###amazon
dt=0.1
steps=2000
T_start=32.9
Tth=34.7
dTex=0.2
Te=34
dtex=80
r=0.0089
s=0.004

T=np.array([i*dt for i in range(steps)])

v=np.zeros((steps,))
Tl=np.zeros((steps,))
g=np.zeros((steps,))

Tf=amazon.T_develop2(dt,T_start,Tth,dTex,Te,dtex,r,s,steps)
amazon.Runge_Kutta_dynamics(v,Tl,g,steps,Tf)
ama_sn=np.diff(Tl)/np.diff(v)

T_amp=1
s_obs1=0.025*1
s_obs2=0.05*1
s_obs3=0.1*1
s_obs4=0.2*1

n_s1=T_amp/s_obs1/np.abs(ama_sn)
n_s2=T_amp/s_obs2/np.abs(ama_sn)
n_s3=T_amp/s_obs3/np.abs(ama_sn)
n_s4=T_amp/s_obs4/np.abs(ama_sn)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(T[100:-1],n_s1[100:],c='r',label='0.025')
ax.plot(T[100:-1],n_s2[100:],c='m',label='0.05')
ax.plot(T[100:-1],n_s3[100:],c='g',label='0.1')
ax.plot(T[100:-1],n_s4[100:],c='b',label='0.2')
plt.legend()
fig.savefig('../../output/amazon/scenario1/noise_signal_ratio2.png')
fig.clf()
plt.close()

print(ama_sn[100:])
with open('../data/amazon/signal_ratio.csv','w',encoding='utf-8') as f:
    for step in range(steps-2):
        f.write('{},'.format(ama_sn[step]))
    f.write(str(ama_sn[steps-2]))
    f.close()
###amaoc
steps=4000
Tst=15
Tref=16
Te=16.5
dlim=1.5
Tth=18
Fref=1.1
Fth=1.296
s=0.01
dTex=0.8
dtex=350

y=np.zeros((steps,))
y[0]=0.2

mu=6.2**(0.5)
ita=3.17*10**(-5)
V=300*4.5*8250
td=180

T=np.array([i*dt for i in range(steps)])
Ta=AMOC.T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps)
F=AMOC.F_develop(Ta,Fth,Fref,Tth,Tref)
AMOC.Runge_Kutta_dynamics(F,y,mu,steps,dt)
Q=AMOC.salinity_flux_to_flow_strength(mu,y,td,ita,V)

amo_sn=np.diff(Ta)/np.diff(Q)

T_amp=1
s_obs1=0.01*10
s_obs2=0.025*10
s_obs3=0.05*10
s_obs4=0.1*10

n_s1=T_amp/s_obs1/np.abs(amo_sn)
n_s2=T_amp/s_obs2/np.abs(amo_sn)
n_s3=T_amp/s_obs3/np.abs(amo_sn)
n_s4=T_amp/s_obs4/np.abs(amo_sn)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(T[100:-1],n_s1[100:],c='r',label='0.01')
ax.plot(T[100:-1],n_s2[100:],c='m',label='0.025')
ax.plot(T[100:-1],n_s3[100:],c='g',label='0.05')
ax.plot(T[100:-1],n_s4[100:],c='b',label='0.1')
plt.legend()
fig.savefig('../../output/amoc/scenario/noise_signal_ratio2.png')
fig.clf()
plt.close()

print(amo_sn[100:])
with open('../data/amoc/signal_ratio.csv','w',encoding='utf-8') as f:
    for step in range(steps-2):
        f.write('{},'.format(amo_sn[step]))
    f.write(str(amo_sn[steps-2]))
    f.close()

