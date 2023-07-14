import numpy as np
from particle import particle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil

def forest_dieback(pre_v,pre_Tl,pre_g,g0,Topt,beta,gamma,dt,alpha):
    dv=(pre_g*pre_v*(1-pre_v)-gamma*pre_v)*dt
    post_Tl=pre_Tl-alpha*dv
    post_v=pre_v+dv

    post_g=g0*(1-((post_Tl-Topt)/beta)**2)
    return post_v,post_Tl

def simple_model(pre_v,pre_Tl,g,gamma,dt,alpha):
    ##ignoring g dynamics
    dv=(g*pre_v*(1-pre_v)-gamma*pre_v)*dt
    post_Tl=pre_Tl-alpha*dv
    post_v=pre_v+dv

    l=len(post_v)

    post_v+=dt*0.05*np.random.randn(l)
    post_Tl+=dt*0.5*np.random.randn(l)

    return post_v,post_Tl
def dynamics(v,Tl,g,steps,Tf):
    ##set hyperparams    alpha=5
    beta=10
    Topt=28
    g0=2
    gamma=0.2
    dt=0.01
    alpha=5
    
    v[0]=0.2
    Tl[0]=Tf+(1-v[0])*alpha
    g[0]=g0*(1-((Tl[0]-Topt)/beta)**2)
    for step in range(steps-1):
        dv=(g[step]*v[step]*(1-v[step])-gamma*v[step])*dt
        Tl[step+1]=Tl[step]-alpha*dv
        v[step+1]=v[step]+dv

        g[step+1]=g0*(1-((Tl[step+1]-Topt)/beta)**2)


if __name__=='__main__':
    print('start simulation')
    ##set initial condition
    print("make obs data")
    ##make true data of the simulation
    Tf=33
    steps=1000
    dt=0.1
    Tl_obs=np.zeros((steps,))
    v_obs=np.zeros((steps,))
    g_obs=np.zeros((steps,))
    dynamics(v_obs,Tl_obs,g_obs,steps,Tf)
    v_nonnoise=v_obs.copy()
    obs_steps=[t*50 for t in range(15)]
    s=0.03
    v_obs+=np.random.randn(steps)*s

    print('finish making observation data')

    print('start filtering prediction')
    print('set initial condition')
    ##set simple model initial conditions 
    alpha=5
    beta=10
    gamma=0.2
    dt=0.01

    s_num=1000
    pcls=particle.ParticleFilter(s_num)
    g_ind=0
    v_ind=1
    Tl_ind=2
    pre_v_ind=3
    pre_Tl_ind=4

    v_results=np.zeros((pcls.n_particle,steps))
    Tl_results=np.zeros((pcls.n_particle,steps))
    g_results=np.zeros((pcls.n_particle,steps))

    v_results[:,0]=0.2
    Tl_results[:,0]=Tf+alpha*(1-v_results[:,0])

    ##self.particleを生成
    pcls.random_sampling()
    
    pcls.particle[:,v_ind]=0.2
    pcls.particle[:,Tl_ind]=Tf+(1-pcls.particle[:,v_ind])*alpha
    
    print('start time extention')
    T=np.array([t*dt+dt for t in range(steps)])


    for step in tqdm(range(steps)):
        ##obs stepsは観測がうったタイムステップ
        pcls.particle[:,pre_v_ind]=pcls.particle[:,v_ind]
        pcls.particle[:,pre_Tl_ind]=pcls.particle[:,Tl_ind]

        pcls.particle[:,v_ind],pcls.particle[:,Tl_ind]=simple_model(
            pcls.particle[:,pre_v_ind],
            pcls.particle[:,pre_Tl_ind],
            pcls.particle[:,g_ind],
            gamma,dt,alpha
            )
        
        if step in obs_steps:
            weights=pcls.norm_likelihood(v_obs[step],pcls.particle[:,v_ind],s)
            inds=pcls.resampling(weights)
            pcls.particle=pcls.particle[inds,:]

        if step%100==0:
            dir='./output/particle/time'+str(step)
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
            pcls.hist_particle(dir)
            
        v_results[:,step]+=pcls.particle[:,v_ind]
        Tl_results[:,step]+=pcls.particle[:,Tl_ind]
        g_results[:,step]+=pcls.particle[:,g_ind]

    print('write graphs')

    fig = plt.figure(figsize=(10,6))

    ax1 = fig.add_subplot(2,2,1)
    ax1.set_xlim(0,dt*steps)
    ax1.set_xlabel('yr')
    ax1.set_ylim(0,1)
    ax1.set_ylabel('TP ratio')

    ax2 = fig.add_subplot(2,2,2)
    ax2.set_xlim(0,dt*steps)
    ax2.set_xlabel('yr')
    ax2.set_ylim(20,45)
    ax2.set_xlabel('Tl(degree Celcius)')

    ax3 = fig.add_subplot(2,2,3)
    ax3.set_xlim(20,45)
    ax3.set_xlabel('Tl(degree Celcius)')
    ax3.set_ylim(0,1)
    ax3.set_ylabel('TP_ratio')

    ax4=fig.add_subplot(2,2,4)
    ax4.set_xlim(0,dt*steps)
    ax4.set_xlabel('yr')
    ax4.set_ylim(0,2)
    ax4.set_ylabel('g')

    ax1.plot(T,np.max(v_results,axis=0),color='gray')
    ax1.plot(T,np.min(v_results,axis=0),color='gray')
    ax1.scatter(T[obs_steps],v_obs[obs_steps],color='blue')
    ax1.plot(T,v_nonnoise,color='black')

    ax2.plot(T,np.max(Tl_results,axis=0),color='gray')
    ax2.plot(T,np.min(Tl_results,axis=0),color='gray')
    ax2.plot(T,Tl_obs,color='black')

    ax3.plot(np.max(Tl_results,axis=0),np.min(v_results,axis=0),color='gray')
    ax3.plot(np.min(Tl_results,axis=0),np.max(v_results,axis=0),color='gray')
    ax3.plot(Tl_obs,v_nonnoise,color='black')

    ax4.plot(T,np.max(g_results,axis=0),color='gray')
    ax4.plot(T,np.min(g_results,axis=0),color='gray')
    ax4.plot(T,g_obs,color='black')

    ax1.plot(T,v_results.mean(axis=0),color='red')
    ax2.plot(T,Tl_results.mean(axis=0),color='red')
    ax3.plot(Tl_results.mean(axis=0),v_results.mean(axis=0),color='red')
    ax4.plot(T,g_results.mean(axis=0),color='red')

    fig.savefig('./output/particle/fig_Tf_'+str(Tf)+'_2.png')