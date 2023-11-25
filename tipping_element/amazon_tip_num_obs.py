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



if __name__=='__main__':
    with open('../data/amazon/tip_num.csv', 'w', encoding='utf-8') as f:
        steps=10000 #steps to execute
        dt=0.1
        #mu=np.array([mu0+mu1*i*dt for i in range(steps)])
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
        '''[0.0089,0.004,0.01,60,'r'],
        [0.0089,0.005,0.2,80,'m'],
        [0.0089,0.004,0.8,300,'r'],'''

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
        for n_ex,s_obs in [[8,0.025],[9,0.05],[10,0.1],[11,0.2]]:
            print('obs_noise:'+str(s_obs))
            s_li=0
            s_li+=s_obs
            print('likelihood sd:'+str(s_li))
            for roop in range(1,7,1):
                obs_num=int(10*(2**roop)/2+1) ####Number of observation
                #obs_num=0
                #roop=3
                print('obs_num:'+str(obs_num))
                fs=int(200/(2**roop)*2)
                if roop==5:
                    obs_num=101
                    fs=20
                if roop==6:
                    obs_num=201
                    fs=10
                for seed in range(1,101,1):
                    #seed=3
                    np.random.seed(seed)

                    Tl_obs=np.zeros((steps,))
                    v_obs=np.zeros((steps,))
                    g_obs=np.zeros((steps,))
                    Runge_Kutta_dynamics(v_obs,Tl_obs,g_obs,steps,Tf,epsilon=epsilon)
                    v_nonnoise=v_obs.copy()
                    #print('smoothing steps:'+str(r_obs*fs))
                    v_obs+=np.random.randn(steps)*s_obs
                    ##set simple model initial conditions 
                    alpha=5
                    beta=10
                    #gamma=0.2
                    #dt=0.1

                    pcls=particle.ParticleFilter(s_num)

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

                    #output='./output/particle/result'+str(n_ex)+'_'+str(roop)
                    #os.mkdir(output)

                    print('the number of observation is '+str(obs_num))
                    #obs_steps=[t*fs for t in range(obs_num)]
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
                    
                    pcls.particle[:,v_ind]=0.8
                    pcls.particle[:,Tl_ind]=T_start+(1-pcls.particle[:,v_ind])*alpha
                    
                    pcls.particle[:,v2_ind]=0.8
                    pcls.particle[:,Tl2_ind]=T_start+(1-pcls.particle[:,v2_ind])*alpha

                    pcls.particle[:,v3_ind]=0.8
                    pcls.particle[:,Tl3_ind]=T_start+(1-pcls.particle[:,v3_ind])*alpha
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
                        '''
                        pcls.particle[:,v_ind],pcls.particle[:,Tl_ind]=simple_model(
                            pcls.particle[:,pre_v_ind],
                            pcls.particle[:,pre_Tl_ind],
                            pcls.particle[:,g_ind],
                            pcls.particle[:,gamma_ind]
                            ,dt,alpha
                            )
                            '''
                        
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
                        '''
                        if step in obs_steps:
                            weights=pcls.norm_likelihood(v_obs[step],pcls.particle[:,v_ind],s_li)
                            inds=pcls.resampling(weights)
                            pcls.particle=pcls.particle[inds,:]

                            pcls.gaussian_inflation(st_inds,a=0.1)
                        '''

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

                    num=len(pcls.particle[pcls.particle[:,v_ind]<0.1])
                    f.write('{},{},{},{}\n'.format(1,s_obs,obs_num,num))
                    print('num of tipping particles:'+str(num))
                    num2=len(pcls.particle[pcls.particle[:,v2_ind]<0.1])
                    f.write('{},{},{},{}\n'.format(2,s_obs,obs_num,num2))
                    print('num2 of tipping particles:'+str(num2))
                    num3=len(pcls.particle[pcls.particle[:,v3_ind]<0.1])
                    f.write('{},{},{},{}\n'.format(3,s_obs,obs_num,num3))
                    print('num3 of tipping particles:'+str(num3))
                    #tip_num.append(num/s_num)
                    #print('Tf at the last observation step : '+str(Tf[obs_steps[-1]]))
        # ログファイルを移動
        #shutil.move('./output.log', output)    
