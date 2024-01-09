import numpy as np
import sys
sys.path.append("../particle_filter/")
import particle
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
    print('temperature noise :'+str(amp*dt**0.5))

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
    yth=0.99
    y_ini=0.2
    ##experimental settings
    n_ex=4
    steps=10000
    obs_num=21
    fs=100 ##observation assimilation frequency
    s_num=1000
    print('obs_num:'+str(obs_num))
    print('s_num:'+str(s_num))
    roop=7
    s_obs=0.025*10 ##observation noise
    s_li=0.025*10 ##likelihood variation
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
    tip_ind=7
    tip_step_ind=8

    y2_ind=9
    pre_y2_ind=10
    Q2_ind=11
    tip2_ind=12
    tip_step2_ind=13

    y3_ind=14
    pre_y3_ind=15
    Q3_ind=16
    tip3_ind=17
    tip_step3_ind=18

    p_num=7 ##dimension of the parameter set

    st_inds=[ita_ind,V_ind,mu_ind,td_ind] ##list of stochastic parameters
    output='../../output/amoc/experitment'+str(n_ex)+'_'+str(roop)
    os.mkdir(output)

    gap=200

    obs_steps=[int(gap/dt)+t*fs for t in range(obs_num)] ##observation steps
    ##initial condition
    pcls.random_sampling_amoc()
    ##set results arrays
    y_results=np.zeros((pcls.n_particle,steps))
    Q_results=np.zeros((pcls.n_particle,steps))
    mu_results=np.zeros((pcls.n_particle,steps))
    V_results=np.zeros((pcls.n_particle,steps))
    ita_results=np.zeros((pcls.n_particle,steps))

    y2_results=np.zeros((pcls.n_particle,steps))
    Q2_results=np.zeros((pcls.n_particle,steps))

    y3_results=np.zeros((pcls.n_particle,steps))
    Q3_results=np.zeros((pcls.n_particle,steps))

    y_results[:,0]=y_ini ##set initial condition
    Q_results[:,0]=salinity_flux_to_flow_strength(
        pcls.particle[:,mu_ind],
        y_results[:,0],
        pcls.particle[:,td_ind],
        pcls.particle[:,ita_ind],
        pcls.particle[:,V_ind]
    )

    y2_results[:,0]=y_ini ##set initial condition
    Q2_results[:,0]=salinity_flux_to_flow_strength(
        pcls.particle[:,mu_ind],
        y2_results[:,0],
        pcls.particle[:,td_ind],
        pcls.particle[:,ita_ind],
        pcls.particle[:,V_ind]
    )
    y3_results[:,0]=y_ini ##set initial condition
    Q3_results[:,0]=salinity_flux_to_flow_strength(
        pcls.particle[:,mu_ind],
        y3_results[:,0],
        pcls.particle[:,td_ind],
        pcls.particle[:,ita_ind],
        pcls.particle[:,V_ind]
    )
    
    pcls.particle[:,y_ind]=y_ini
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
    Tst=13.5
    Tref=16
    dlim=1.5
    Tth=18
    dTex=0.8
    r=(Tth-Tst)/402
    Te=Tst+dlim
    s=0.01
    dtex=350
    print('tip time:'+str((Tth-Tst)/r))
    Fref=1.1
    Fth=1.296
    T=np.array([dt*i for i in range(steps)])
    #Ta=T_develop(steps,dt,mu_arr,Tst,dlim,mu0)
    Ta=T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=1.5)
    F=F_develop(Ta,Fth,Fref,Tth,Tref)

    dTex=0.8
    r=(Tth-Tst)/402
    Te=Tst+dlim
    s=0.01
    dtex=300
    print('tip time:'+str((Tth-Tst)/r))
    Ta2=T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=1.5)
    F2=F_develop(Ta2,Fth,Fref,Tth,Tref)

    dTex=0.01
    r=0.0089
    Tst=Tth-402*r
    Te=Tst+dlim
    s=0.01
    dtex=40
    print('tip time:'+str((Tth-Tst)/r))
    Ta3=T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=1.5)
    F3=F_develop(Ta3,Fth,Fref,Tth,Tref)

    ##make observation data
    y_obs=np.zeros((steps,))
    y2_obs=np.zeros((steps,))
    y3_obs=np.zeros((steps,))
    y_obs[0]=y_ini
    y2_obs[0]=y_ini
    y3_obs[0]=y_ini

    Runge_Kutta_dynamics(F,y_obs,mu,steps,dt)
    Runge_Kutta_dynamics(F2,y2_obs,mu,steps,dt)
    Runge_Kutta_dynamics(F3,y3_obs,mu,steps,dt)

    Q_obs=salinity_flux_to_flow_strength(mu,y_obs,td,ita,V)
    Q2_obs=salinity_flux_to_flow_strength(mu,y2_obs,td,ita,V)
    Q3_obs=salinity_flux_to_flow_strength(mu,y3_obs,td,ita,V)

    Q_obs_nonnoise=Q_obs.copy()
    Q2_obs_nonnoise=Q2_obs.copy()
    Q3_obs_nonnoise=Q3_obs.copy()

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
        pcls.particle[:,Q_ind]=salinity_flux_to_flow_strength(
            pcls.particle[:,mu_ind],
            pcls.particle[:,y_ind],
            pcls.particle[:,td_ind],
            pcls.particle[:,ita_ind],
            pcls.particle[:,V_ind]
        )

        pcls.particle[:,pre_y2_ind]=pcls.particle[:,y2_ind]

        pcls.particle[:,y2_ind]=simple_model(
            pcls.particle[:,pre_y2_ind],
            step,F2,
            pcls.particle[:,mu_ind],
            dt
        )
        pcls.particle[:,Q2_ind]=salinity_flux_to_flow_strength(
            pcls.particle[:,mu_ind],
            pcls.particle[:,y2_ind],
            pcls.particle[:,td_ind],
            pcls.particle[:,ita_ind],
            pcls.particle[:,V_ind]
        )

        pcls.particle[:,pre_y3_ind]=pcls.particle[:,y3_ind]

        pcls.particle[:,y3_ind]=simple_model(
            pcls.particle[:,pre_y3_ind],
            step,F3,
            pcls.particle[:,mu_ind],
            dt
        )
        pcls.particle[:,Q3_ind]=salinity_flux_to_flow_strength(
            pcls.particle[:,mu_ind],
            pcls.particle[:,y3_ind],
            pcls.particle[:,td_ind],
            pcls.particle[:,ita_ind],
            pcls.particle[:,V_ind]
        )

        if step in obs_steps:
            weights=pcls.norm_likelihood(
                Q_obs[step],
                pcls.particle[:,Q_ind],
                s_li
            )
            dir=output+'/time'+str(step)
            ##describe the results pictures
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
            inds=pcls.resampling(weights)
            pcls.particle=pcls.particle[inds,:]
            pcls.hist_particle(dir,p_num)

            pcls.gaussian_inflation(st_inds,a=0.1)

        pcls.particle[:,tip_step_ind]+=((pcls.particle[:,y_ind]>yth)-pcls.particle[:,tip_ind])*step*(1-pcls.particle[:,tip_ind])
        pcls.particle[:,tip_ind]=(pcls.particle[:,tip_step_ind]>0)

        pcls.particle[:,tip_step2_ind]+=((pcls.particle[:,y2_ind]>yth)-pcls.particle[:,tip2_ind])*step*(1-pcls.particle[:,tip2_ind])
        pcls.particle[:,tip2_ind]=(pcls.particle[:,tip_step2_ind]>0)

        pcls.particle[:,tip_step3_ind]+=((pcls.particle[:,y3_ind]>yth)-pcls.particle[:,tip3_ind])*step*(1-pcls.particle[:,tip3_ind])
        pcls.particle[:,tip3_ind]=(pcls.particle[:,tip_step3_ind]>0)
        ##add reults to the result array
        y_results[:,step]+=pcls.particle[:,y_ind]
        Q_results[:,step]+=pcls.particle[:,Q_ind]
        mu_results[:,step]+=pcls.particle[:,mu_ind]
        V_results[:,step]+=pcls.particle[:,V_ind]
        ita_results[:,step]+=pcls.particle[:,ita_ind]

        y2_results[:,step]+=pcls.particle[:,y2_ind]
        Q2_results[:,step]+=pcls.particle[:,Q2_ind]  

        y3_results[:,step]+=pcls.particle[:,y3_ind]
        Q3_results[:,step]+=pcls.particle[:,Q3_ind] 

    ##Write result map
    fig=plt.figure(figsize=(10,9))
    view_steps=steps

    ax1 = fig.add_subplot(3,2,1)
    ax1.set_xlim(0,dt*view_steps)
    ax1.set_ylim(0,30)
    ax1.set_xlabel('yr')
    ax1.set_ylabel('Q')

    ax2 = fig.add_subplot(3,2,2)
    ax2.set_xlim(0,dt*view_steps)
    ax2.set_xlabel('yr')
    ax2.set_xlabel('y')
    ax2.set_xlim(0,dt*steps)
    ax2.set_ylim(0,1.5)

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

    ax6=fig.add_subplot(3,2,6)
    ax6.set_xlim(0,dt*view_steps)
    ax6.set_xlabel('yr')
    ax6.set_ylabel('temperature')

    ax1.plot(T,np.max(Q_results,axis=0),color='gray',linestyle='solid')
    ax1.plot(T,np.min(Q_results,axis=0),color='gray',linestyle='solid')
    ax1.scatter(T[obs_steps],Q_obs_nonnoise[obs_steps],color='blue',linestyle='solid')
    ax1.plot(T,Q_obs_nonnoise,color='black',linestyle='solid')

    ax1.plot(T,np.max(Q2_results,axis=0),color='gray',linestyle='dashed')
    ax1.plot(T,np.min(Q2_results,axis=0),color='gray',linestyle='dashed')
    ax1.plot(T,Q2_obs_nonnoise,color='black',linestyle='dashed')

    ax1.plot(T,np.max(Q3_results,axis=0),color='gray',linestyle='dashdot')
    ax1.plot(T,np.min(Q3_results,axis=0),color='gray',linestyle='dashdot')
    ax1.plot(T,Q3_obs_nonnoise,color='black',linestyle='dashdot')
    ax1.vlines(obs_num*fs*dt+gap,0, 10, color='g', linestyle='dotted')

    ax2.plot(T,np.max(y_results,axis=0),color='gray',linestyle='solid')
    ax2.plot(T,np.min(y_results,axis=0),color='gray',linestyle='solid')
    ax2.plot(T,y_obs,color='black',linestyle='solid')

    ax2.plot(T,np.max(y2_results,axis=0),color='gray',linestyle='dashed')
    ax2.plot(T,np.min(y2_results,axis=0),color='gray',linestyle='dashed')
    ax2.plot(T,y2_obs,color='black',linestyle='dashed')

    ax2.plot(T,np.max(y3_results,axis=0),color='gray',linestyle='dashdot')
    ax2.plot(T,np.min(y3_results,axis=0),color='gray',linestyle='dashdot')
    ax2.plot(T,y3_obs,color='black',linestyle='dashdot')

    ax2.vlines(obs_num*fs*dt+gap,0,30,color='g', linestyle='dotted')

    ax3.plot(T,np.min(ita_results,axis=0),color='gray',linestyle='solid')
    ax3.plot(T,np.max(ita_results,axis=0),color='gray',linestyle='solid')
    ax3.plot(T,[ita for i in range(steps)],color='black',linestyle='solid')
    ax3.vlines(obs_num*fs*dt+gap, 0,0.0001,color='g',linestyle='dotted')

    #ax4.plot(T,np.max(V_results,axis=0),color='gray')
    #ax4.plot(T,np.min(V_results,axis=0),color='gray')
    #ax4.plot(T,[V for i in range(steps)],color='black')
    ax4.hlines(Fth,0,view_steps,color='g',linestyle='dotted')
    ax4.vlines(obs_num*fs*dt+gap, 0.5, 2.5, color='g', linestyle='dotted')

    ax6.plot(T,Ta,color='black',linestyle='solid')
    ax6.plot(T,Ta2,color='black',linestyle='dashed')
    ax6.plot(T,Ta3,color='black',linestyle='dashdot')
    ax6.hlines(Tth,0,view_steps,color='g',linestyle='dotted')
    ax6.vlines(obs_num*fs*dt+gap, 15, 25, color='g', linestyle='dotted')

    import csv

    with open('../data/amoc/amoc_stable.csv') as f:
        ans=csv.reader(f)
        ans=np.array(list(ans)).astype(float)

    ax5.scatter(ans[:,0],ans[:,1],s=5,color='black')

    ax5.plot(F,y_obs,c='blue')

    ax1.plot(T,Q_results.mean(axis=0),color='red',linestyle='solid')
    ax1.plot(T,Q2_results.mean(axis=0),color='red',linestyle='dashed')
    ax1.plot(T,Q3_results.mean(axis=0),color='red',linestyle='dashdot')

    ax2.plot(T,y_results.mean(axis=0),color='red',linestyle='solid')
    ax2.plot(T,y2_results.mean(axis=0),color='red',linestyle='dashed')
    ax2.plot(T,y3_results.mean(axis=0),color='red',linestyle='dashdot')

    ax3.plot(T,ita_results.mean(axis=0),color='red',linestyle='solid')

    ax4.plot(T,F,color='black',linestyle='solid')
    ax4.plot(T,F2,color='black',linestyle='dashed')
    ax4.plot(T,F3,color='black',linestyle='dashdot')

    ax5.plot(F,y_results.mean(axis=0),color='red',linestyle='solid')
    ax5.plot(F,y2_results.mean(axis=0),color='red',linestyle='dashed')
    ax5.plot(F,y3_results.mean(axis=0),color='red',linestyle='dashdot')
    fig.savefig(output+'/result.png')
    fig.clf()

    ##scenatio1
    tip_time=np.zeros((pcls.n_particle,4))
    tip_time+=pcls.particle[:,[ita_ind,mu_ind,tip_step_ind,tip_ind]]
    tip_time[:,2]+=(1-tip_time[:,3])*steps

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(tip_time[:,2],bins=20)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1000)
    ax.set_title('tipping time distribution')
    fig.savefig(output+'/tipping_time_distribution.png')
    fig.clf()
    plt.close()
    
    ##tipping time
    x = tip_time[:,0]
    y = tip_time[:,1]
    value=tip_time[:,2]
    eff_ind=tip_time[:,2]<steps
    ineff_ind=tip_time[:,2]==steps
    print(value)        
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    # カラーマップを生成
    cm = plt.cm.get_cmap('RdYlBu')
    mappable = ax.scatter(x[eff_ind], y[eff_ind], c=value[eff_ind],vmin=0,vmax=5000, cmap=cm,s=1)
    ax.set_title('tipping_area')
    ax.set_xlabel('ita')
    ax.set_ylabel('mu')
    ax.set_xlim(10**(-5),5*10**(-5))
    ax.set_ylim(0.5,3.5)
    fig.colorbar(mappable,ax=ax)
    ax.scatter(x[ineff_ind],y[ineff_ind],color='black',s=1)
    fig.savefig(f"{output}/tipping_heatmap.png")
    fig.clf()
    plt.close()

    ##scenatio2
    tip_time=np.zeros((pcls.n_particle,4))
    tip_time+=pcls.particle[:,[ita_ind,mu_ind,tip_step2_ind,tip2_ind]]
    tip_time[:,2]+=(1-tip_time[:,3])*steps

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(tip_time[:,2],bins=20)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1000)
    ax.set_title('tipping time distribution')
    fig.savefig(output+'/tipping_time_distribution2.png')
    fig.clf()
    plt.close()
    
    ##tipping time
    x = tip_time[:,0]
    y = tip_time[:,1]
    value=tip_time[:,2]
    eff_ind=tip_time[:,2]<steps
    ineff_ind=tip_time[:,2]==steps
    print(value)        
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    # カラーマップを生成
    cm = plt.cm.get_cmap('RdYlBu')
    mappable = ax.scatter(x[eff_ind], y[eff_ind], c=value[eff_ind],vmin=0,vmax=5000, cmap=cm,s=1)
    ax.set_title('tipping_area')
    ax.set_xlabel('ita')
    ax.set_ylabel('mu')
    ax.set_xlim(10**(-5),5*10**(-5))
    ax.set_ylim(0.5,3.5)
    fig.colorbar(mappable,ax=ax)
    ax.scatter(x[ineff_ind],y[ineff_ind],color='black',s=1)
    fig.savefig(f"{output}/tipping_heatmap2.png")
    fig.clf()
    plt.close()

    ##scenatio3
    tip_time=np.zeros((pcls.n_particle,4))
    tip_time+=pcls.particle[:,[ita_ind,mu_ind,tip_step3_ind,tip3_ind]]
    tip_time[:,2]+=(1-tip_time[:,3])*steps

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(tip_time[:,2],bins=20)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1000)
    ax.set_title('tipping time distribution')
    fig.savefig(output+'/tipping_time_distribution3.png')
    fig.clf()
    plt.close()
    
    ##tipping time
    x = tip_time[:,0]
    y = tip_time[:,1]
    value=tip_time[:,2]
    eff_ind=tip_time[:,2]<steps
    ineff_ind=tip_time[:,2]==steps
    print(value)        
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    # カラーマップを生成
    cm = plt.cm.get_cmap('RdYlBu')
    mappable = ax.scatter(x[eff_ind], y[eff_ind], c=value[eff_ind],vmin=0,vmax=5000, cmap=cm,s=1)
    ax.set_title('tipping_area')
    ax.set_xlabel('ita')
    ax.set_ylabel('mu')
    ax.set_xlim(10**(-5),5*10**(-5))
    ax.set_ylim(0.5,3.5)
    fig.colorbar(mappable,ax=ax)
    ax.scatter(x[ineff_ind],y[ineff_ind],color='black',s=1)
    fig.savefig(f"{output}/tipping_heatmap3.png")
    fig.clf()
    plt.close()

    shutil.move('./output.log', output) 
