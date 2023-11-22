import numpy as np
from particle import particle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil
def T_develop(steps,dt,mu,T_start,dTlim,mu0):
    ##Tstart is the temprature shown as Tf in preindustrial period in Amazon forest
    ##dTlim is the param which shows the goal determined in Paris conference
    dT0=0.89
    beta=0.0215
    gamma=beta-mu0*(dTlim-dT0)

    t=np.array([i*dt for i in range(steps)])

    T=T_start+beta*t*(t<dT0/beta)+dT0*(t>=dT0/beta)+gamma*t-(1-np.exp(mu*t*(-1)))*(gamma*t-(dTlim-dT0))
    
    return T

def T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps):
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

    amp=0
    T+=amp*np.random.randn(steps)*dt
    print('temperature noise :'+str(amp*dt))

    return T

def AMOC(F,mu,y):
    k=F-y*(1+(mu*(1-y))*(mu*(1-y)))
    return k

def salinity_flux_to_flow_strength(mu,y,td,ita,V):
    Q=ita*V*(1+(mu*(1-y))*(mu*(1-y)))/td
    return Q

def Runge_Kutta_dynamics(F,y,mu,steps,dt):
    for step in range(steps-1):
        k1=AMOC(F[step],mu,y[step])
        k2=AMOC(F[step],mu,y[step]+k1*dt/2)
        k3=AMOC(F[step],mu,y[step]+k2*dt/2)
        k4=AMOC(F[step],mu,y[step]+k3*dt/2)

        y[step+1]=y[step]+dt/6*(k1+k2+k3+k4)

def F_develop(T,Fth,Fref,Tth,Tref):
    ##forcing parameter about salinity
    F=Fref+(T-Tref)/(Tth-Tref)*(Fth-Fref)
    return F

def simple_model(pre_y,step,F,mu,dt):
    k1=AMOC(F[step],mu,pre_y)
    k2=AMOC(F[step],mu,pre_y+k1*dt/2)
    k3=AMOC(F[step],mu,pre_y+k2*dt/2)
    k4=AMOC(F[step],mu,pre_y+k3*dt/2)

    y=pre_y+dt/6*(k1+k2+k3+k4)

    return y

if __name__=='__main__':
    ##experimental settings
    n_ex=1
    steps=15000
    obs_num=0
    s_num=1000
    print('obs_num:'+str(obs_num))
    print('s_num:'+str(s_num))
    roop=3
    s_obs=0.1 ##observation noise
    s_li=0.1 ##likelihood variation
    dt=0.1 ##time step
    pcls=particle.ParticleFilter(s_num)

    ##set particles indexes
    ita_ind=0
    V_ind=1
    mu_ind=2
    td_ind=3

    y_ind=4
    pre_y_ind=5
    Q_ind=6

    p_num=7 ##dimension of the parameter set

    st_inds=[] ##list of stochastic parameters
    output='./output/amoc/experitment'+str(n_ex)+'_'+str(roop)
    os.mkdir(output)

    fs=10 ##observation assimilation frequency
    obs_steps=[t*fs for t in range(obs_num)] ##observation steps
    ##initial condition
    pcls.random_sampling_amoc()
    ##set results arrays
    y_results=np.zeros((pcls.n_particle,steps))
    Q_results=np.zeros((pcls.n_particle,steps))
    mu_results=np.zeros((pcls.n_particle,steps))
    V_results=np.zeros((pcls.n_particle,steps))
    ita_results=np.zeros((pcls.n_particle,steps))

    y_results[:,0]=0.35 ##set initial condition
    Q_results[:,0]=salinity_flux_to_flow_strength(
        pcls.particle[:,mu_ind],
        y_results[:,0],
        pcls.particle[:,td_ind],
        pcls.particle[:,ita_ind],
        pcls.particle[:,V_ind]
    )

    pcls.particle[:,y_ind]=0.35
    pcls.particle[:,Q_ind]=salinity_flux_to_flow_strength(
        pcls.particle[:,mu_ind],
        pcls.particle[:,y_ind],
        pcls.particle[:,td_ind],
        pcls.particle[:,ita_ind],
        pcls.particle[:,V_ind]        
    )
    ##set tue param
    mu=6.2**(0.5)
    ita=3.17*10**(-5)
    V=300*4.5*8250
    td=180

    ##determine F divelopment
    Tst=10
    Tref=25
    dlim=10
    Tth=29
    dTex=1
    Te=Tst+dlim
    r=0.3
    s=0.005
    dtex=300

    Fref=0.8
    Fth=1.3
    T=np.array([dt*i for i in range(steps)])
    #Ta=T_develop(steps,dt,mu_arr,Tst,dlim,mu0)
    Ta=T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps)
    F=F_develop(Ta,Fth,Fref,Tth,Tref)
    ##make observation data
    y_obs=np.zeros((steps,))
    y_obs[0]=0.35

    Runge_Kutta_dynamics(F,y_obs,mu,steps,dt)
    Q_obs=salinity_flux_to_flow_strength(mu,y_obs,td,ita,V)
    Q_obs_nonnoise=Q_obs.copy()
    Q_obs+=np.random.randn(steps)*s_obs*dt**(0.5)
    ##start data assimilation
    for step in range(steps):
        pcls.particle[:,pre_y_ind]=pcls.particle[:,y_ind]

        pcls.particle[:,y_ind]=simple_model(
            pcls.particle[:,pre_y_ind],
            step,F,
            pcls.particle[:,mu_ind],
            dt
        )

        if step in obs_steps:
            weights=pcls.norm_likelihood(
                Q_obs,
                pcls.particle[:,Q_ind],
                s_li
            )
            dir=output+'/time'+str(step)
            ##describe the results pictures
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)

            pcls.hist_particle(dir,p_num)

            pcls.gaussian_inflation(st_inds,a=0.1)

        ##add reults to the result array
        y_results[:,step]+=pcls.particle[:,y_ind]
        Q_results[:,step]=salinity_flux_to_flow_strength(
            pcls.particle[:,mu_ind],
            y_results[:,step],
            pcls.particle[:,td_ind],
            pcls.particle[:,ita_ind],
            pcls.particle[:,V_ind]
        )
        mu_results[:,step]+=pcls.particle[:,mu_ind]
        V_results[:,step]+=pcls.particle[:,V_ind]
        ita_results[:,step]+=pcls.particle[:,ita_ind]
    
    ##Write result map
    fig=plt.figure(figsize=(10,9))
    view_steps=steps

    ax1 = fig.add_subplot(3,2,1)
    ax1.set_xlim(0,dt*view_steps)
    ax1.set_xlabel('yr')
    ax1.set_ylabel('Q')

    ax2 = fig.add_subplot(3,2,2)
    ax2.set_xlim(0,dt*view_steps)
    ax2.set_xlabel('yr')
    ax2.set_xlabel('y')

    ax3=fig.add_subplot(3,2,3)
    ax3.set_xlim(0,dt*view_steps)
    ax3.set_xlabel('yr')
    ax3.set_ylabel('ita')

    ax4=fig.add_subplot(3,2,4)
    ax4.set_xlim(0,dt*view_steps)
    ax4.set_xlabel('yr')
    ax4.set_ylabel('F')

    ax5=fig.add_subplot(3,2,5)
    ax5.set_xlabel('F')
    ax5.set_ylabel('y steady')


    ax1.plot(T,np.max(Q_results,axis=0),color='gray')
    ax1.plot(T,np.min(Q_results,axis=0),color='gray')
    ax1.scatter(T[obs_steps],Q_obs_nonnoise[obs_steps],color='blue')
    ax1.plot(T,Q_obs_nonnoise,color='black')
    ax1.vlines(obs_num,0, 10, color='g', linestyles='dotted')

    ax2.plot(T,np.max(y_results,axis=0),color='gray')
    ax2.plot(T,np.min(y_results,axis=0),color='gray')
    ax2.plot(T,y_obs,color='black')
    ax2.vlines(obs_num,0,10,color='g', linestyles='dotted')

    ax3.plot(T,np.min(ita_results,axis=0),color='gray')
    ax3.plot(T,np.max(ita_results,axis=0),color='gray')
    ax3.plot(T,[ita for i in range(steps)],color='black')
    ax3.vlines(obs_num, 0,0.0001,color='g',linestyles='dotted')

    #ax4.plot(T,np.max(V_results,axis=0),color='gray')
    #ax4.plot(T,np.min(V_results,axis=0),color='gray')
    #ax4.plot(T,[V for i in range(steps)],color='black')
    ax4.hlines(Fth,0,view_steps,color='g',linestyles='dotted')
    ax4.vlines(obs_num, 0.5, 2.5, color='g', linestyles='dotted')

    #ax6.plot(T,F,color='black')

    import csv

    with open('/Users/amane/Downloads/research/sawada/thesis/tipping_points/coding/output/amoc/amoc_stable.csv') as f:
        ans=csv.reader(f)
        ans=np.array(list(ans)).astype(float)

    ax5.scatter(ans[:,0],ans[:,1],s=5,color='black')

    ax5.plot(F,y_obs,c='blue')
    ax1.plot(T,Q_results.mean(axis=0),color='red')
    ax2.plot(T,y_results.mean(axis=0),color='red')
    ax3.plot(T,ita_results.mean(axis=0),color='red')
    ax4.plot(T,F,color='black')
    ax5.plot(F,y_results.mean(axis=0),color='red')
    fig.savefig(output+'/result.png')
    fig.clf()

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(T,Ta)
    fig.savefig(output+'/temperature_profile.png')
    fig.clf()
    shutil.move('./output.log', output) 
