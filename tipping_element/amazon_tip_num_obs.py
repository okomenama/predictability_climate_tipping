import numpy as np
import sys
sys.path.append("../particle_filter")
import particle
import amazon
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil
import yaml

amp=os.environ['AMPLITUDE']
output=os.environ['EXAMAZONOUTDIR']
ex_num=os.environ['EXNUM']
red=os.environ['REDCHAFRE']

amp=int(amp)
red = int(red)

with open(ex_num +'.yml') as file:
    config = yaml.safe_load(file.read())

accuracy_list = config['accuracy']
obs_fs_list = config['obs_fs']

print('accuracy_list : ')
print(accuracy_list)

print('observation number and interval')
print(obs_fs_list)

with open(output+'/'+ ex_num+'/tip_num_amp'+str(amp)+'.csv', 'w', encoding='utf-8') as f:
    steps=10000 #steps to execute
    dt=0.1
    tip_num=[]
    s_num=1000 ##number of particles
    print('s_num:'+str(s_num))
    r_obs=0
    epsilon=1 ####time scale param
    #mu=np.array([mu0+mu1*i*dt for i in range(steps)])
    T_start=32.7
    Tth=34.7
    Te=34
    dTex=0.8
    dtex=350
    r=(Tth-T_start)/202
    s=0.008
    Tf=amazon.T_develop2(dt,T_start,Tth,dTex,Te,dtex,r,s,steps,amp=amp)

    dTex2=-0.1
    dtex2=-10
    r2=(Tth-T_start)/202
    s2=0.008
    np.random.seed(1)

    Tf2=amazon.T_develop2(dt,T_start,Tth,dTex2,Te,dtex2,r2,s2,steps,amp=amp)
    for s_obs in accuracy_list:
        print('obs_noise:'+str(s_obs))
        s_li=0
        s_li+=s_obs
        for obs_num,fs in obs_fs_list:
            print('obs_num:'+str(obs_num))
            weights = np.ones(s_num)
            for seed in tqdm(range(1,101,1)):
                #seed=3
                np.random.seed(seed)

                Tl_obs=np.zeros((steps,))
                v_obs=np.zeros((steps,))
                g_obs=np.zeros((steps,))
                amazon.Runge_Kutta_dynamics(v_obs,Tl_obs,g_obs,steps,Tf,epsilon=epsilon)
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


                tip_ind=11
                tip_step_ind=12
                tip2_ind=19
                tip2_step_ind=20

                p_num=5

                st_inds=[g0_ind,Topt_ind,gamma_ind,alpha_ind,beta_ind]

                #output='./output/particle/result'+str(n_ex)+'_'+str(roop)
                #os.mkdir(output)

                print('the number of observation is '+str(obs_num))
                obs_steps=[(t+1)*fs for t in range(obs_num)]

                g0_results=np.zeros((pcls.n_particle,steps))

                v_results=np.zeros((pcls.n_particle,steps))
                v2_results=np.zeros((pcls.n_particle,steps))

                Tl_results=np.zeros((pcls.n_particle,steps))
                Tl2_results=np.zeros((pcls.n_particle,steps))

                g_results=np.zeros((pcls.n_particle,steps))
                g2_results=np.zeros((pcls.n_particle,steps))

                gamma_results=np.zeros((pcls.n_particle,steps))

                v_results[:,0]=0.8
                Tl_results[:,0]=T_start+alpha*(1-v_results[:,0])

                ##self.particleを生成
                pcls.random_sampling()
                
                pcls.particle[:,v_ind]=0.8
                pcls.particle[:,Tl_ind]=T_start+(1-pcls.particle[:,v_ind])*alpha
                
                pcls.particle[:,v2_ind]=0.8
                pcls.particle[:,Tl2_ind]=T_start+(1-pcls.particle[:,v2_ind])*alpha

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
                    '''
                    pcls.particle[:,v_ind],pcls.particle[:,Tl_ind]=simple_model(
                        pcls.particle[:,pre_v_ind],
                        pcls.particle[:,pre_Tl_ind],
                        pcls.particle[:,g_ind],
                        pcls.particle[:,gamma_ind]
                        ,dt,alpha
                        )
                        '''
                    
                    pcls.particle[:,v_ind],pcls.particle[:,Tl_ind],pcls.particle[:,g_ind]=amazon.simple_model4(
                        pcls.particle[:,pre_v_ind],
                        pcls.particle[:,pre_Tl_ind],
                        pcls.particle[:,pre_g_ind],
                        pcls.particle[:,g0_ind],
                        beta,
                        pcls.particle[:,Topt_ind],
                        pcls.particle[:,gamma_ind]
                        ,dt,alpha,Tf,step,epsilon=epsilon
                        )
                    pcls.particle[:,v2_ind],pcls.particle[:,Tl2_ind],pcls.particle[:,g2_ind]=amazon.simple_model4(
                        pcls.particle[:,pre_v2_ind],
                        pcls.particle[:,pre_Tl2_ind],
                        pcls.particle[:,pre_g2_ind],
                        pcls.particle[:,g0_ind],
                        beta,
                        pcls.particle[:,Topt_ind],
                        pcls.particle[:,gamma_ind]
                        ,dt,alpha,Tf2,step,epsilon=epsilon
                        )
                
                    if step in obs_steps:
                        weights*=pcls.norm_likelihood(v_obs[step],pcls.particle[:,v_ind],s_li)

                        if red == 1:

                            if step%50 == 0:
                                inds=pcls.resampling(weights)
                                pcls.particle=pcls.particle[inds,:]

                                pcls.gaussian_inflation(st_inds,a=0.1)

                                weights = np.ones(s_num)
                        else:
                            inds=pcls.resampling(weights)
                            pcls.particle=pcls.particle[inds,:]

                            pcls.gaussian_inflation(st_inds,a=0.1)

                            weights = np.ones(s_num)
            

                    pcls.particle[:,tip_step_ind]+=((pcls.particle[:,v_ind]<0.01)-pcls.particle[:,tip_ind])*step*(1-pcls.particle[:,tip_ind])
                    pcls.particle[:,tip_ind]=(pcls.particle[:,tip_step_ind]>0)

                    pcls.particle[:,tip2_step_ind]+=((pcls.particle[:,v2_ind]<0.01)-pcls.particle[:,tip2_ind])*step*(1-pcls.particle[:,tip2_ind])
                    pcls.particle[:,tip2_ind]=(pcls.particle[:,tip2_step_ind]>0)

                    v_results[:,step]+=pcls.particle[:,v_ind]
                    Tl_results[:,step]+=pcls.particle[:,Tl_ind]
                    g_results[:,step]+=pcls.particle[:,g_ind]
                    gamma_results[:,step]+=pcls.particle[:,gamma_ind]
                    g0_results[:,step]+=pcls.particle[:,g0_ind]

                    v2_results[:,step]+=pcls.particle[:,v2_ind]
                    Tl2_results[:,step]+=pcls.particle[:,Tl2_ind]
                    g2_results[:,step]+=pcls.particle[:,g2_ind]

                num=len(pcls.particle[pcls.particle[:,v_ind]<0.1])
                f.write('{},{},{},{}\n'.format(1,s_obs,obs_num,num))

                num2=len(pcls.particle[pcls.particle[:,v2_ind]<0.1])
                f.write('{},{},{},{}\n'.format(2,s_obs,obs_num,num2))

