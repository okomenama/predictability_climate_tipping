import numpy as np
import sys
sys.path.append("../particle_filter")
import particle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil

##function setting
def T_develop(steps,dt,mu,T_start,dTlim,mu0):
    ##Tstart is the temprature shown as Tf in preindustrial period in Amazon forest
    ##dTlim is the param which shows the goal determined in Paris conference
    dT0=0.89
    beta=0.0128
    gamma=beta-mu0*(dTlim-dT0)

    t=np.array([i*dt for i in range(steps)])

    T=T_start+dT0+gamma*t-(1-np.exp(mu*t*(-1)))*(gamma*t-(dTlim-dT0))
    
    return T

def T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps):
    ##This is simpler version of temperature profile
    ##Set dummy threshold d1,d2,d3
    d1=(Tth+dTex-Tst)/r
    d2=d1+dtex-dTex/s-dTex/r
    d3=d2+(Tth+dTex-Te)/s

    t=np.array([i*dt for i in range(steps)])
    T=(Tst+r*t)*(t<d1) \
    +(Tth+dTex)*(t>d1)*(t<d2) \
    +(Tth+dTex-s*(t-d2))*(t<d3)*(t>d2) \
    +Te*(t>d3)

    amp=0
    T+=amp*np.random.randn(steps)*dt
    print('temperature noise :'+str(amp*dt))

    return T

def MCMC_resampling():
    return

def forest_dieback(pre_v,pre_Tl,pre_g,g0,Topt,beta,gamma,dt,alpha):
    epsilon=0.5
    print('time scale(large means slow varying) : '+str(epsilon))
    dv=(pre_g*pre_v*(1-pre_v)-gamma*pre_v)*dt/epsilon
    post_Tl=pre_Tl-alpha*dv
    post_v=pre_v+dv

    post_g=g0*(1-((post_Tl-Topt)/beta)**2)
    return post_v,post_Tl

def simple_model(pre_v,pre_Tl,g,gamma,dt,alpha):
    ##ignoring g Euler_dynamics
    dv=(g*pre_v*(1-pre_v)-gamma*pre_v)*dt
    post_Tl=pre_Tl-alpha*dv
    post_v=pre_v+dv

    l=len(post_v)

    post_v+=dt*0.01*np.random.randn(l)
    post_Tl+=dt*0.03*np.random.randn(l)

    return post_v,post_Tl

def simple_model2(pre_v,pre_Tl,pre_g,g0,beta,Topt,gamma,dt,alpha):
    ##ignoring g Euler_dynamics
    dv=(pre_g*pre_v*(1-pre_v)-gamma*pre_v)*dt
    post_Tl=pre_Tl-alpha*dv
    post_v=pre_v+dv

    l=len(post_v)

    post_v+=dt*0.05*np.random.randn(l)
    post_Tl+=dt*0.5*np.random.randn(l)

    post_g=g0*(1-(post_Tl-Topt)/beta)*(1+(post_Tl-Topt)/beta)

    return post_v,post_Tl,post_g

def simple_model3(pre_v,pre_Tl,pre_g,g0,beta,Topt,gamma,dt,alpha,Tf,step):
    ##ignoring g Euler_dynamics
    dv=(pre_g*pre_v*(1-pre_v)-gamma*pre_v)*dt
    dTf=Tf[step]-Tf[step-1]

    post_Tl=pre_Tl-alpha*dv+dTf
    post_v=pre_v+dv

    l=len(post_v)
    noise=0.01*np.random.randn(l)
    post_v+=dt*noise
    post_Tl-=alpha*dt*noise

    post_g=g0*(1-(post_Tl-Topt)/beta)*(1+(post_Tl-Topt)/beta)

    return post_v,post_Tl,post_g

def simple_model4(pre_v,pre_Tl,pre_g,g0,beta,Topt,gamma,dt,alpha,Tf,step,epsilon=1):
    ##Runge Kutta
    l=len(pre_v)

    k1=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],pre_v)
    k2=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],pre_v+dt/2*k1)
    k3=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],pre_v+dt/2*k2)
    k4=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],pre_v+dt*k3)

    post_v=pre_v+dt/6*(k1+2*k2+2*k3+k4)/epsilon
    #post_v+=dt*0.05*np.random.randn(l)

    post_Tl=Tf[step]+alpha*(1-post_v)
    post_g=g0*(1-(post_Tl-Topt)/beta)*(1+(post_Tl-Topt)/beta)

    return post_v,post_Tl,post_g

def back_dynamics(v,Tl,g,g0,beta,Topt,gamma,dt,alpha,Tf,step,epsilon=1):
    k1=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v)
    k2=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v-dt/2*k1)
    k3=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v-dt/2*k2)
    k4=amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf[step],v-dt*k3)

    v=v-dt/6*(k1+2*k2+2*k3+k4)/epsilon
    Tl=Tf[step+1]+alpha*(1-v)
    g=g0*(1-(Tl-Topt)/beta)*(1+(Tl-Topt)/beta)

def likelihood(v_obs,v,s):
    return np.exp((-1)*(v-v_obs)*(v-v_obs)/2/s/s)/np.sqrt(2*np.pi)/s

def resampling(l,weights):
    weights /= weights.sum()
    w_cumsum = np.cumsum(weights)
    k_list = np.searchsorted(w_cumsum, np.random.uniform(0,1,size = l))
    return k_list

def smoother(v,Tl,g,g0,beta,Topt,gamma,dt,alpha,Tf,fs,step,r_obs,s,epsilon=1):
    l=len(v)
    rv=v.copy()
    rTl=Tl.copy()
    rg=g.copy()
    rg0=g0.copy()
    rTopt=Topt.copy()


    likelihood_vec=np.ones(l)
    rsteps=r_obs*fs
    rmin=step-rsteps
    robs_steps=[rmin+i*fs for i in range(r_obs)]

    for rstep in range(rsteps):
        back_dynamics(rv,rTl,rg,rg0,beta,rTopt,gamma,dt,alpha,Tf,step-rstep)
        if step-rstep in robs_steps:
            likelihood_vec*=likelihood(v_obs[step-rstep],rv,s)

    return likelihood_vec



def Euler_dynamics(v,Tl,g,steps,Tf,dt=0.1):
    ##set hyperparams    alpha=5
    beta=10
    Topt=28
    g0=2
    gamma=0.2
    alpha=5
    
    v[0]=0.8
    Tl[0]=Tf[0]+(1-v[0])*alpha
    g[0]=g0*(1-((Tl[0]-Topt)/beta)**2)

    for step in range(steps-1):
        dv=(g[step]*v[step]*(1-v[step])-gamma*v[step])*dt
        dTf=Tf[step+1]-Tf[step]
        Tl[step+1]=Tl[step]-alpha*dv+dTf
        v[step+1]=v[step]+dv

        g[step+1]=g0*(1-((Tl[step+1]-Topt)/beta)**2)

def amazon_forest_dieoff(g0,Topt,gamma,alpha,beta,Tf,v):
    g=g0*(1-(((Tf+alpha*(1-v))-Topt)/beta)**2)
    k=g*v*(1-v)-gamma*v

    return k

def Runge_Kutta_dynamics(v,Tl,g,steps,Tf,iters=1,dt=0.1,epsilon=1,noise=0,amp=0.05):
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
        if noise==1:
            ns=np.random.randn(iters)*dt*amp
            v[step+1]=v[step]+dt/6*(k1+2*k2+2*k3+k4)/epsilon+ns
            Tl[step+1]=Tf[step+1]+alpha*(1-v[step+1])
            g[step+1]=g0*(1-((Tl[step+1]-Topt)/beta)**2)
        else:
            v[step+1]=v[step]+dt/6*(k1+2*k2+2*k3+k4)/epsilon
            Tl[step+1]=Tf[step+1]+alpha*(1-v[step+1])
            g[step+1]=g0*(1-((Tl[step+1]-Topt)/beta)**2)

def mutual_information(pri_particles,post_particles,##state space variables, n+1D array(params+state variable)
                  bin_width, bin_num, dim,min, #width of discretization of state space n+1D array
                  n_particles
                  ):
    
    pri_bin_ind=np.zeros((n_particles,dim))
    post_bin_ind=np.zeros((n_particles,dim))

    pri_bin_count=np.zeros(tuple(bin_num))
    post_bin_count=np.zeros(tuple(bin_num))


    for i in range(dim):
        pri_inds=np.digitize(pri_particles[:,i],bins=[min[i]+j*bin_width[i] for j in range(1,bin_num[i],1)])
        pri_bin_ind[:,i]+=pri_inds 

        post_inds=np.digitize(post_particles[:,i],bins=[min[i]+j*bin_width[i] for j in range(1,bin_num[i],1)])
        post_bin_ind[:,i]+=post_inds 
    
    for i in range(n_particles):
        pri_bin_count[tuple([[int(pri_bin_ind[i,j])] for j in range(dim)])]+=1
        post_bin_count[tuple([[int(post_bin_ind[i,j])] for j in range(dim)])]+=1

    Hprior=(-1)*np.sum(np.where(pri_bin_count <=0,0,pri_bin_count/n_particles*np.log(pri_bin_count/n_particles)))
    Hpost=(-1)*np.sum(np.where(post_bin_count<=0,0,post_bin_count/n_particles*np.log(post_bin_count/n_particles)))
    #print('Hprior:'+str(Hprior))
    #print('Hpost:'+str(Hpost))
    return Hprior-Hpost



if __name__=='__main__':
    n_ex=1
    s_obs=0.025
    roop=4
    steps=10000 #steps to execute
    dt=0.1
    #mu=np.array([mu0+mu1*i*dt for i in range(steps)])
    obs_num=int(10*(2**roop)/2+1) ####Number of observation
    fs=int(200/(2**roop)*2)
    if roop==5:
        obs_num=101
        fs=20
    if roop==6:
        obs_num=201
        fs=10
    print('obs_num:'+str(obs_num))
    '''
    T_start=32.9
    dTlim=1.5
    Tf=T_develop(steps,dt,mu,T_start,dTlim,mu0)
    '''
    T_start=32.9
    Tth=34.7
    dTex=0.2
    Te=34
    dtex=80
    r=0.0089
    s=0.004
    a=0.1
    ##Time when temperature reached the Tth
    print('T_start:'+str(T_start))
    print('Tth:'+str(Tth))
    print('dTex:'+str(dTex))
    print('Te:'+str(Te))
    print('dtex:'+str(dtex))
    print('r:'+str(r))
    print('s:'+str(s))
    s_num=1000 ##number of particles
    print('s_num:'+str(s_num))

    r_obs=0

    #cm = plt.cm.get_cmap('RdYlBu_r')

    print('start simulation')
    ##set initial condition
    print("make obs data")
    #make true data of the simulation

    ##make Tf as time varying
    '''
    mu_array=np.array([
    [-0.012,0.00015],
    [-0.01,0.000135],
    [-0.008,0.000125],
    [-0.005,0.00012],
    [-0.003,0.00011],
    [-0.001,0.0001],
    [0.001,0.00009],
    [0.003,0.00007],
    [0.01,0.00003]])
    '''
    epsilon=1 ####time scale param

    mu0=0
    mu1=0.000026

    #for mu0,mu1 in mu_array
    tip_num=[]

    tip_point=(Tth-T_start)/r

    Tf=T_develop2(dt,T_start,Tth,dTex,Te,dtex,r,s,steps)
    s_li=s_obs
    print('likelihood sd:'+str(s_li))
    print('obs_noise:'+str(s_obs))

    seed=20
    np.random.seed(seed)
    dTex2=0.8
    dtex2=300
    r2=0.0089
    s2=0.007
    Tf2=T_develop2(dt,T_start,Tth,dTex2,Te,dtex2,r2,s2,steps)

    dTex3=0.01
    dtex3=60
    r3=0.0089
    s3=0.004
    Tf3=T_develop2(dt,T_start,Tth,dTex3,Te,dtex3,r3,s3,steps)

    Tl_obs=np.zeros((steps,))
    v_obs=np.zeros((steps,))
    g_obs=np.zeros((steps,))
    Runge_Kutta_dynamics(v_obs,Tl_obs,g_obs,steps,Tf,epsilon=epsilon)
    v_nonnoise=v_obs.copy()
    print('smoothing steps:'+str(r_obs*fs))
    v_obs+=np.random.randn(steps)*s_obs
    ##set simple model initial conditions 
    alpha=5
    beta=10
    #gamma=0.2
    #dt=0.1

    pcls=particle.ParticleFilter(s_num)
    pcls_no_obs=particle.ParticleFilter(s_num)
    Topt_ind=0
    g0_ind=1
    gamma_ind=2
    alpha_ind=3
    beta_ind=4

    v_ind=5
    Tl_ind=6
    g_ind=7
    pre_v_ind=8
    pre_Tl_ind=9
    pre_g_ind=10

    v2_ind=13
    Tl2_ind=14
    g2_ind=15
    pre_v2_ind=16
    pre_Tl2_ind=17
    pre_g2_ind=18

    v3_ind=19
    Tl3_ind=20
    g3_ind=21
    pre_v3_ind=22
    pre_Tl3_ind=23
    pre_g3_ind=24

    tip_ind=11
    tip_step_ind=12
    tip2_ind=25
    tip2_step_ind=26
    tip3_ind=27
    tip3_step_ind=28
    p_num=5

    st_inds=[g0_ind,Topt_ind,gamma_ind,alpha_ind,beta_ind]

    output='../../output/amazon/result'+str(n_ex)+'_'+str(roop)
    os.mkdir(output)

    print('the number of observation is '+str(obs_num))
    obs_steps=[t*fs for t in range(obs_num)]
    g0_results=np.zeros((pcls.n_particle,steps))

    v_results=np.zeros((pcls.n_particle,steps))
    v2_results=np.zeros((pcls.n_particle,steps))
    v3_results=np.zeros((pcls.n_particle,steps))

    Tl_results=np.zeros((pcls.n_particle,steps))
    Tl2_results=np.zeros((pcls.n_particle,steps))
    Tl3_results=np.zeros((pcls.n_particle,steps))

    g_results=np.zeros((pcls.n_particle,steps))
    g2_results=np.zeros((pcls.n_particle,steps))
    g3_results=np.zeros((pcls.n_particle,steps))

    gamma_results=np.zeros((pcls.n_particle,steps))

    v_results[:,0]=0.8
    Tl_results[:,0]=T_start+alpha*(1-v_results[:,0])

    ##self.particleを生成
    pcls.random_sampling()
    pcls_no_obs.random_sampling()
    pcls_no_obs.particle[:,:]=pcls.particle[:,:]
    
    pcls.particle[:,v_ind]=0.8
    pcls.particle[:,Tl_ind]=T_start+(1-pcls.particle[:,v_ind])*alpha

    pcls.particle[:,v2_ind]=0.8
    pcls.particle[:,Tl2_ind]=T_start+(1-pcls.particle[:,v2_ind])*alpha

    pcls.particle[:,v3_ind]=0.8
    pcls.particle[:,Tl3_ind]=T_start+(1-pcls.particle[:,v3_ind])*alpha

    pcls_no_obs.particle[:,v_ind]=0.8
    pcls_no_obs.particle[:,Tl_ind]=T_start+(1-pcls_no_obs.particle[:,v_ind])*alpha

    pcls_no_obs.particle[:,v2_ind]=0.8
    pcls_no_obs.particle[:,Tl2_ind]=T_start+(1-pcls_no_obs.particle[:,v2_ind])*alpha

    pcls_no_obs.particle[:,v3_ind]=0.8
    pcls_no_obs.particle[:,Tl3_ind]=T_start+(1-pcls_no_obs.particle[:,v3_ind])*alpha
    
    print('start time devlop')
    T=np.array([t*dt for t in range(steps)])

    for step in range(steps):
        ##obs stepsは観測がうったタイムステップ
        pcls.particle[:,pre_v_ind]=pcls.particle[:,v_ind]
        pcls.particle[:,pre_Tl_ind]=pcls.particle[:,Tl_ind]
        pcls.particle[:,pre_g_ind]=pcls.particle[:,g_ind]

        pcls.particle[:,pre_v2_ind]=pcls.particle[:,v2_ind]
        pcls.particle[:,pre_Tl2_ind]=pcls.particle[:,Tl2_ind]
        pcls.particle[:,pre_g2_ind]=pcls.particle[:,g2_ind]

        pcls.particle[:,pre_v3_ind]=pcls.particle[:,v3_ind]
        pcls.particle[:,pre_Tl3_ind]=pcls.particle[:,Tl3_ind]
        pcls.particle[:,pre_g3_ind]=pcls.particle[:,g3_ind]
        
        pcls.particle[:,v_ind],pcls.particle[:,Tl_ind],pcls.particle[:,g_ind]=simple_model4(
            pcls.particle[:,pre_v_ind],
            pcls.particle[:,pre_Tl_ind],
            pcls.particle[:,pre_g_ind],
            pcls.particle[:,g0_ind],
            beta,
            pcls.particle[:,Topt_ind],
            pcls.particle[:,gamma_ind]
            ,dt,alpha,Tf,step,epsilon=epsilon
            )
        pcls.particle[:,v2_ind],pcls.particle[:,Tl2_ind],pcls.particle[:,g2_ind]=simple_model4(
            pcls.particle[:,pre_v2_ind],
            pcls.particle[:,pre_Tl2_ind],
            pcls.particle[:,pre_g2_ind],
            pcls.particle[:,g0_ind],
            beta,
            pcls.particle[:,Topt_ind],
            pcls.particle[:,gamma_ind]
            ,dt,alpha,Tf2,step,epsilon=epsilon
            )
        pcls.particle[:,v3_ind],pcls.particle[:,Tl3_ind],pcls.particle[:,g3_ind]=simple_model4(
            pcls.particle[:,pre_v3_ind],
            pcls.particle[:,pre_Tl3_ind],
            pcls.particle[:,pre_g3_ind],
            pcls.particle[:,g0_ind],
            beta,
            pcls.particle[:,Topt_ind],
            pcls.particle[:,gamma_ind]
            ,dt,alpha,Tf3,step,epsilon=epsilon
            )
        
        
        pcls_no_obs.particle[:,pre_v_ind]=pcls_no_obs.particle[:,v_ind]
        pcls_no_obs.particle[:,pre_Tl_ind]=pcls_no_obs.particle[:,Tl_ind]
        pcls_no_obs.particle[:,pre_g_ind]=pcls_no_obs.particle[:,g_ind]

        pcls_no_obs.particle[:,pre_v2_ind]=pcls_no_obs.particle[:,v2_ind]
        pcls_no_obs.particle[:,pre_Tl2_ind]=pcls_no_obs.particle[:,Tl2_ind]
        pcls_no_obs.particle[:,pre_g2_ind]=pcls_no_obs.particle[:,g2_ind]

        pcls_no_obs.particle[:,pre_v3_ind]=pcls_no_obs.particle[:,v3_ind]
        pcls_no_obs.particle[:,pre_Tl3_ind]=pcls_no_obs.particle[:,Tl3_ind]
        pcls_no_obs.particle[:,pre_g3_ind]=pcls_no_obs.particle[:,g3_ind]

        '''
        pcls.particle[:,v_ind],pcls.particle[:,Tl_ind]=simple_model(
            pcls.particle[:,pre_v_ind],
            pcls.particle[:,pre_Tl_ind],
            pcls.particle[:,g_ind],
            pcls.particle[:,gamma_ind]
            ,dt,alpha
            )
            '''
        
        pcls_no_obs.particle[:,v_ind],pcls_no_obs.particle[:,Tl_ind],pcls_no_obs.particle[:,g_ind]=simple_model4(
            pcls_no_obs.particle[:,pre_v_ind],
            pcls_no_obs.particle[:,pre_Tl_ind],
            pcls_no_obs.particle[:,pre_g_ind],
            pcls_no_obs.particle[:,g0_ind],
            beta,
            pcls_no_obs.particle[:,Topt_ind],
            pcls_no_obs.particle[:,gamma_ind]
            ,dt,alpha,Tf,step,epsilon=epsilon
            )
        pcls_no_obs.particle[:,v2_ind],pcls_no_obs.particle[:,Tl2_ind],pcls_no_obs.particle[:,g2_ind]=simple_model4(
            pcls_no_obs.particle[:,pre_v2_ind],
            pcls_no_obs.particle[:,pre_Tl2_ind],
            pcls_no_obs.particle[:,pre_g2_ind],
            pcls_no_obs.particle[:,g0_ind],
            beta,
            pcls_no_obs.particle[:,Topt_ind],
            pcls_no_obs.particle[:,gamma_ind]
            ,dt,alpha,Tf2,step,epsilon=epsilon
            )
        pcls_no_obs.particle[:,v3_ind],pcls_no_obs.particle[:,Tl3_ind],pcls_no_obs.particle[:,g3_ind]=simple_model4(
            pcls_no_obs.particle[:,pre_v3_ind],
            pcls_no_obs.particle[:,pre_Tl3_ind],
            pcls_no_obs.particle[:,pre_g3_ind],
            pcls_no_obs.particle[:,g0_ind],
            beta,
            pcls_no_obs.particle[:,Topt_ind],
            pcls_no_obs.particle[:,gamma_ind]
            ,dt,alpha,Tf3,step,epsilon=epsilon
            )
        if step in obs_steps:
            weights=pcls.norm_likelihood(v_obs[step],pcls.particle[:,v_ind],s_li)
            inds=pcls.resampling(weights)
            pcls.particle=pcls.particle[inds,:]
            
            dir=output+'/time'+str(obs_num)+'_'+str(step)
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
            
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            ax.set_xlabel('g0')
            ax.set_ylabel('Topt')
            #ax.set_xlim((0,2.6))
            #ax.set_ylim((26,33))
            #mappable=ax.scatter(un_data[:,0],un_data[:,1], c=fre,cmap=cm)
            ax.scatter(pcls.particle[:,g0_ind],pcls.particle[:,Topt_ind])
            ax.set_title('particle_distribution')
            #fig.colorbar(mappable,ax=ax)
            fig.savefig(dir+'/particle_distribution.png')
            fig.clf()
            plt.close()
            

            pcls.gaussian_inflation(st_inds,a=0.1)
            
            data=pcls.particle[:,st_inds].copy()

            #un_data,fre=np.unique(data,return_counts=True,axis=0)
            #x=pcls.particle[:,g0_ind].copy()
            #y=pcls.particle[:,gamma_ind].copy()
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            ax.set_xlabel('g0')
            ax.set_ylabel('Topt')
            #ax.set_xlim((0,2.6))
            #ax.set_ylim((26,33))
            #mappable=ax.scatter(un_data[:,0],un_data[:,1], c=fre,cmap=cm)
            ax.scatter(data[:,0],data[:,1])
            ax.set_title('particle_distribution')
            #fig.colorbar(mappable,ax=ax)
            fig.savefig(dir+'/particle_distribution_inflated.png')
            fig.clf()
            plt.close()

            if (step>r_obs*fs)&(r_obs>0):
                rweights=smoother(pcls.particle[:,v_ind],
                                pcls.particle[:,Tl_ind],
                                pcls.particle[:,g_ind],
                                pcls.particle[:,g0_ind],
                                beta,
                                pcls.particle[:,Topt_ind],
                                pcls.particle[:,gamma_ind],
                                dt,alpha,Tf,fs,step,r_obs,s_li)
                
                rinds=pcls.resampling(rweights)
                pcls.particle=pcls.particle[rinds,:]
                
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                ax.set_xlabel('g0')
                ax.set_ylabel('Topt')
                #ax.set_xlim((0,2.6))
                #ax.set_ylim((26,33))
                #mappable=ax.scatter(un_data[:,0],un_data[:,1], c=fre,cmap=cm)
                ax.scatter(pcls.particle[:,g0_ind],pcls.particle[:,Topt_ind])
                ax.set_title('particle_distribution')
                #fig.colorbar(mappable,ax=ax)
                fig.savefig(dir+'/particle_distribution_smoothing.png')
                fig.clf()
                
                pcls.gaussian_inflation(st_inds,a)
                
                data=pcls.particle[:,st_inds].copy()

                #un_data,fre=np.unique(data,return_counts=True,axis=0)
                #x=pcls.particle[:,g0_ind].copy()
                #y=pcls.particle[:,gamma_ind].copy()
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                ax.set_xlabel('g0')
                ax.set_ylabel('Topt')
                #ax.set_xlim((0,2.6))
                #ax.set_ylim((26,33))
                #mappable=ax.scatter(un_data[:,0],un_data[:,1], c=fre,cmap=cm)
                ax.scatter(data[:,0],data[:,1])
                ax.set_title('particle_distribution')
                #fig.colorbar(mappable,ax=ax)
                fig.savefig(dir+'/particle_distribution_smoothing_inflated.png')
                fig.clf()
                
            
            pcls.hist_particle(dir,p_num)
            
        pcls.particle[:,tip_step_ind]+=((pcls.particle[:,v_ind]<0.01)-pcls.particle[:,tip_ind])*step*(1-pcls.particle[:,tip_ind])
        pcls.particle[:,tip_ind]=(pcls.particle[:,tip_step_ind]>0)

        pcls.particle[:,tip2_step_ind]+=((pcls.particle[:,v2_ind]<0.01)-pcls.particle[:,tip2_ind])*step*(1-pcls.particle[:,tip2_ind])
        pcls.particle[:,tip2_ind]=(pcls.particle[:,tip2_step_ind]>0)

        pcls.particle[:,tip3_step_ind]+=((pcls.particle[:,v3_ind]<0.01)-pcls.particle[:,tip3_ind])*step*(1-pcls.particle[:,tip3_ind])
        pcls.particle[:,tip3_ind]=(pcls.particle[:,tip3_step_ind]>0)

        v_results[:,step]+=pcls.particle[:,v_ind]
        Tl_results[:,step]+=pcls.particle[:,Tl_ind]
        g_results[:,step]+=pcls.particle[:,g_ind]
        gamma_results[:,step]+=pcls.particle[:,gamma_ind]
        g0_results[:,step]+=pcls.particle[:,g0_ind]

        v2_results[:,step]+=pcls.particle[:,v2_ind]
        Tl2_results[:,step]+=pcls.particle[:,Tl2_ind]
        g2_results[:,step]+=pcls.particle[:,g2_ind]

        v3_results[:,step]+=pcls.particle[:,v3_ind]
        Tl3_results[:,step]+=pcls.particle[:,Tl3_ind]
        g3_results[:,step]+=pcls.particle[:,g3_ind]
    
    print('write graphs')

    ##結果画像をここに描く
    fig = plt.figure(figsize=(10,9))
    view_steps=10000

    ax1 = fig.add_subplot(3,2,1)
    ax1.set_xlim(0,dt*view_steps)
    ax1.set_xlabel('yr')
    ax1.set_ylim(0,1)
    ax1.set_ylabel('TP ratio')

    ax2 = fig.add_subplot(3,2,2)
    ax2.set_xlim(0,dt*view_steps)
    ax2.set_xlabel('yr')
    ax2.set_ylim(20,45)
    ax2.set_xlabel('Tl(degree Celcius)')

    ax3=fig.add_subplot(3,2,3)
    ax3.set_xlim(0,dt*view_steps)
    ax3.set_xlabel('yr')
    ax3.set_ylim(-2,3)
    ax3.set_ylabel('g0')

    ax4=fig.add_subplot(3,2,4)
    ax4.set_xlim(0,dt*view_steps)
    ax4.set_xlabel('yr')
    ax4.set_ylim(-2,3)
    ax4.set_ylabel('g')

    ax5=fig.add_subplot(3,2,5)
    ax5.set_xlim(28,38)
    ax5.set_xlabel('yr')
    ax5.set_ylabel('TP ratio')
    ax5.set_ylim(-1,1.1)

    ax6=fig.add_subplot(3,2,6)
    ax6.set_xlim(0,dt*view_steps)
    ax6.set_xlabel('yr')
    ax6.set_ylim(0,1)
    ax6.set_ylabel('gamma')
    

    ax1.plot(T,np.max(v_results,axis=0),color='gray')
    ax1.plot(T,np.min(v_results,axis=0),color='gray')
    ax1.scatter(T[obs_steps],v_obs[obs_steps],color='blue')
    ax1.plot(T,v_nonnoise,color='black')
    ax1.vlines(obs_num, 0, 1, color='g', linestyles='dotted')
    
    ax2.plot(T,np.max(Tl_results,axis=0),color='gray')
    ax2.plot(T,np.min(Tl_results,axis=0),color='gray')
    ax2.plot(T,Tl_obs,color='black')
    ax2.vlines((obs_num-1)*fs*dt, 30, 40, color='g', linestyles='dotted')

    ax3.plot(T,np.max(g0_results,axis=0),color='gray')
    ax3.plot(T,np.min(g0_results,axis=0),color='gray')
    ax3.plot(T,[2.0 for i in range(len(T))],color='black')
    ax3.vlines((obs_num-1)*fs*dt, 0.5, 2.5, color='g', linestyles='dotted')

    ax4.plot(T,np.max(g_results,axis=0),color='gray')
    ax4.plot(T,np.min(g_results,axis=0),color='gray')
    ax4.plot(T,g_obs,color='black')
    ax4.vlines((obs_num-1)*fs*dt, 0.5, 2.5, color='g', linestyles='dotted')

    import csv

    with open('../data/amazon/stable_points.csv') as f:
        ans=csv.reader(f)
        ans=np.array(list(ans)).astype(float)

    ax5.scatter(ans[:,0],ans[:,1],s=5,color='green')



    ax6.plot(T,np.max(gamma_results,axis=0),color='gray')
    ax6.plot(T,np.min(gamma_results,axis=0),color='gray')
    ax6.plot(T,[0.2 for i in range(len(Tf))],color='black')
    ax6.vlines((obs_num-1)*fs*dt, 0, 1, color='g', linestyles='dotted')
    

    ax1.plot(T,v_results.mean(axis=0),color='red') 
    ax2.plot(T,Tl_results.mean(axis=0),color='red')
    ax3.plot(T,g0_results.mean(axis=0),color='red')
    ax4.plot(T,g_results.mean(axis=0),color='red')
    ax5.plot(Tf,v_results.mean(axis=0),color='red')
    ax6.plot(T,gamma_results.mean(axis=0),color='red')
    
    num=len(pcls.particle[pcls.particle[:,v_ind]<0.1])
    print('num of tipping particles:'+str(num))

    num2=len(pcls.particle[pcls.particle[:,v2_ind]<0.1])
    print('num of tipping particles2:'+str(num2))

    num3=len(pcls.particle[pcls.particle[:,v3_ind]<0.1])
    print('num of tipping particles3:'+str(num3))
    #tip_num.append(num/s_num)
    #print('Tf at the last observation step : '+str(Tf[obs_steps[-1]]))
    
    fig.savefig(output+'/fig_forslide_Tf_tipping_tmv'+str(obs_num)+'.png')
    fig.clf()
    plt.close()

    print(' ')
    ##温度のプロファイルを図示する
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(0,dt*view_steps)
    ax.set_ylim(30,40)
    ax.vlines((obs_num-1)*fs*dt,ymin=30,ymax=40,linestyles='dashdot')
    ax.hlines(Tth,0,1000,linestyles='dotted')
    ax.plot(T,Tf,color='black')
    fig.savefig(output+'/Tf_profile.png')
    fig.clf()
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(0,dt*view_steps)
    ax.set_ylim(30,40)
    ax.vlines((obs_num-1)*fs*dt,ymin=30,ymax=40,linestyles='dashdot')
    ax.hlines(Tth,0,1000,linestyles='dotted')
    ax.plot(T,Tf2,color='black')
    fig.savefig(output+'/Tf2_profile.png')
    fig.clf()
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(0,dt*view_steps)
    ax.set_ylim(30,40)
    ax.vlines((obs_num-1)*fs*dt,ymin=30,ymax=40,linestyles='dashdot')
    ax.hlines(Tth,0,1000,linestyles='dotted')
    ax.plot(T,Tf3,color='black')
    fig.savefig(output+'/Tf3_profile.png')
    fig.clf()
    plt.close()
    #roop+=1
    ###最後に生き残っている粒子のtippingまでの時間をカラーマップとして二次元平面上にプロットする
    tip_time=np.zeros((pcls.n_particle,4))
    tip_time+=pcls.particle[:,[g0_ind,Topt_ind,tip_step_ind,tip_ind]]
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
    mappable = ax.scatter(x[eff_ind], y[eff_ind], c=value[eff_ind],vmin=0,vmax=3000, cmap=cm)
    ax.set_title('tipping_area')
    ax.set_xlabel('g0')
    ax.set_ylabel('Topt')
    ax.set_xlim(0,3)
    ax.set_ylim(25,41)
    fig.colorbar(mappable,ax=ax)
    ax.scatter(x[ineff_ind],y[ineff_ind],color='black')
    fig.savefig(f"{output}/tipping_heatmap.png")
    fig.clf()
    plt.close()

    tip_time=np.zeros((pcls.n_particle,4))
    tip_time+=pcls.particle[:,[g0_ind,Topt_ind,tip2_step_ind,tip2_ind]]
    tip_time[:,2]+=(1-tip_time[:,3])*steps

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(tip_time[:,2],bins=20)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1000)
    ax.set_title('tipping time distribution')
    fig.savefig(output+'/tipping2_time_distribution.png')
    fig.clf()
    plt.close()

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
    mappable = ax.scatter(x[eff_ind], y[eff_ind], c=value[eff_ind],vmin=0,vmax=3000, cmap=cm)
    ax.set_title('tipping_area')
    ax.set_xlabel('g0')
    ax.set_ylabel('Topt')
    ax.set_xlim(0,3)
    ax.set_ylim(25,41)
    fig.colorbar(mappable,ax=ax)
    ax.scatter(x[ineff_ind],y[ineff_ind],color='black')
    fig.savefig(f"{output}/tipping_heatmap2.png")
    fig.clf()
    plt.close()

    tip_time=np.zeros((pcls.n_particle,4))
    tip_time+=pcls.particle[:,[g0_ind,Topt_ind,tip3_step_ind,tip3_ind]]
    tip_time[:,2]+=(1-tip_time[:,3])*steps  

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(tip_time[:,2],bins=20)
    ax.set_xlim(0,10000)
    ax.set_ylim(0,1000)
    ax.set_title('tipping time distribution')
    fig.savefig(output+'/tipping3_time_distribution.png')
    fig.clf()
    plt.close()

    x = tip_time[:,0]
    y = tip_time[:,1]
    value=tip_time[:,2]
    eff_ind=tip_time[:,2]<steps
    ineff_ind=tip_time[:,2]==steps
       
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    # カラーマップを生成
    cm = plt.cm.get_cmap('RdYlBu')
    mappable = ax.scatter(x[eff_ind], y[eff_ind], c=value[eff_ind],vmin=0,vmax=3000, cmap=cm)
    ax.set_title('tipping_area')
    ax.set_xlabel('g0')
    ax.set_ylabel('Topt')
    ax.set_xlim(0,3)
    ax.set_ylim(25,41)
    fig.colorbar(mappable,ax=ax)
    ax.scatter(x[ineff_ind],y[ineff_ind],color='black')
    fig.savefig(f"{output}/tipping_heatmap3.png")
    fig.clf()
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(pcls.particle[:,v_ind],bins=20,range=(0,1))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1000)
    ax.set_xlabel('tropical forest ratio')
    ax.set_title('histgram of particle')
    fig.savefig(output+'/v_result.png')
    fig.clf()
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(pcls.particle[:,v2_ind],bins=20,range=(0,1))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1000)
    ax.set_xlabel('tropical forest ratio')
    ax.set_title('histgram of particle')
    fig.savefig(output+'/v2_result.png')
    fig.clf()
    plt.close()

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(pcls.particle[:,v3_ind],bins=20,range=(0,1))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1000)
    ax.set_xlabel('tropical forest ratio')
    ax.set_title('histgram of particle')
    fig.savefig(output+'/v3_result.png')
    fig.clf()
    plt.close()
    #print(pcls.particle[:,v_ind])

    min=[0,25,1.0]
    iters=70
    mi1_list=np.zeros((iters,))
    mi2_list=np.zeros((iters,))
    mi3_list=np.zeros((iters,))

    for i in range(1,iters+1,1):
        bin_num=[i,i,i]
        bin_width=[1/i,10/i,2/i]
        print('v range:({},{})'.format(min[0],min[0]+bin_num[0]*bin_width[0]))
        print('g0 range:({},{})'.format(min[1],min[1]+bin_num[1]*bin_width[1]))
        print('Topt range:({},{})'.format(min[2],min[2]+bin_num[2]*bin_width[2]))
        
        mi1=mutual_information(
            pcls_no_obs.particle[:,[v_ind,g0_ind,Topt_ind]],
            pcls.particle[:,[v_ind,g0_ind,Topt_ind]],
            bin_width,
            bin_num,
            3,
            min,
            pcls.n_particle
        )
        #print('シナリオ1 I(Z;Ya|X)=H(Z|X)-H(Z|Y,X)='+str(mi1))
        mi2=mutual_information(
            pcls_no_obs.particle[:,[v2_ind,g0_ind,Topt_ind]],
            pcls.particle[:,[v2_ind,g0_ind,Topt_ind]],
            bin_width,
            bin_num,
            3,
            min,
            pcls.n_particle
        )
        #print('シナリオ2 I(Z;Ya|X)=H(Z|X)-H(Z|Y,X)='+str(mi2))
        mi3=mutual_information(
            pcls_no_obs.particle[:,[v3_ind,g0_ind,Topt_ind]],
            pcls.particle[:,[v3_ind,g0_ind,Topt_ind]],
            bin_width,
            bin_num,
            3,
            min,
            pcls.n_particle
        )
        mi1_list[i-1]+=mi1
        mi2_list[i-1]+=mi2
        mi3_list[i-1]+=mi3

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(-1,2.4)
    ax.plot([i+1 for i in range(iters)],mi1_list,color='r',label='0.2deg, 80years')
    ax.plot([i+1 for i in range(iters)],mi2_list,color='g',label='0.8deg, 300years')
    ax.plot([i+1 for i in range(iters)],mi3_list,color='b',label='0.01deg, 60years')
    ax.set_xlabel('num of bin')
    ax.set_ylabel('mutual information')
    ax.set_title('mutual information change')
    fig.legend()
    fig.savefig(output+'/mutual_information.png')
    fig.clf()
    plt.close()

    #print('シナリオ3 I(Z;Ya|X)=H(Z|X)-H(Z|Xa)='+str(mi3))
    # ログファイルを移動
    shutil.move('./output.log', output)    
