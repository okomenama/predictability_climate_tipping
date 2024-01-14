import sys
import AMOC
import matplotlib.pyplot as plt
import numpy as np
import shutil
##make amoc scenario
steps=10000
dt=0.1
Tth=18
Te=16.5
Tref=16
Fth=1.296
Fref=1.1
mu=6.2**0.5
ita=3.17*10**(-5)
V=300*4.5*8250
td=180

Tst=15
params=[
    #[Tst,(Tth-Tst)/402,0.010,0.8,350,'r']
    #,[Tst,(Tth-Tst)/402,0.005,0.2,100,'g']
    [Tst,(Tth-Tst)/402,0.005,-0.1,-10,'b']
]
T=np.array([t*dt for t in range(steps)])
output='../../output/amoc'
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax1.set_xlim(0,dt*steps)
ax1.set_ylim(13,23)
ax1.set_ylabel('Ta')
ax1.hlines(Tth,0,1000,linestyles='dotted')

ax2=fig.add_subplot(2,1,2)
ax2.set_xlim(0,dt*steps)
ax2.set_ylim(0,15)
ax2.set_ylabel('Q')
ax2.set_xlabel('time(years)')
for T_start,r,s,dTex,dtex,c in params:
    y_obs=np.zeros((steps,))
    epsilon=1
    np.random.seed(2)
    Ta=AMOC.T_develop2(dt,T_start,Tth,dTex,Te,dtex,r,s,steps,amp=4)
    F=AMOC.F_develop(Ta,Fth,Fref,Tth,Tref)
    AMOC.Runge_Kutta_dynamics(F,y_obs,mu,steps,dt)
    Q_obs=AMOC.salinity_flux_to_flow_strength(mu,y_obs,td,ita,V)
    ax1.plot(T,Ta,color=c)
    ax2.plot(T,Q_obs,color=c)
ax1.set_title('temperature & Q profile')
fig.savefig(output+'/scinareo1_amp4.png')
fig.clf()