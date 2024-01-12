import numpy as np
import sys
sys.path.append("../particle_filter/")
import particle
import AMOC
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import shutil

if __name__=='__main__':
    yth=0.99
    with open('../data/final_result/amoc/tip_num_freq_amp4.csv','w',encoding='utf-8') as f:
        steps=10000
        dt=0.1
        ##set Temperature profile
        Tst=15
        Tref=16
        Te=16.5
        dlim=1.5
        Tth=18
        Fref=1.1
        Fth=1.296
        ##set tue param
        mu=6.2**(0.5)
        ita=3.17*10**(-5)
        V=300*4.5*8250
        td=180
        s_num=1000 ##number of particles
        print('s_num:'+str(s_num))

        r=(Tth-Tst)/402
        s=0.01
        dTex=0.8
        dtex=350
        T=np.array([dt*i for i in range(steps)])     

        Ta=AMOC.T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=4)
        F=AMOC.F_develop(Ta,Fth,Fref,Tth,Tref)
        ##other scenarios
        dTex=0.2
        s=0.005
        dtex=100
        Ta2=AMOC.T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=4)
        F2=AMOC.F_develop(Ta2,Fth,Fref,Tth,Tref)

        dTex=0.01
        r=(Tth-Tst)/402
        s=0.005
        dtex=40
        #np.random.seed(1)
        Ta3=AMOC.T_develop2(dt,Tst,Tth,dTex,Te,dtex,r,s,steps,amp=4)
        F3=AMOC.F_develop(Ta3,Fth,Fref,Tth,Tref)

        y_ini=0.2
        for r_obs in [0.01,0.025,0.05,0.1]:
            s_obs=r_obs*10
            print('obs_noise:'+str(s_obs))
            s_li=0
            s_li+=s_obs
            '''
            for obs_num,fs in [[11,200],[21,100],
                               [41,50],[81,25],
                               [101,20],[201,10]]:
            '''

            for obs_num,fs in [[90,20],[80,20],[70,20]]:
                print('obs_num:'+str(obs_num))
                print('fs:'+str(fs))
                for seed in range(1,101,1):
                    gap=200
                    np.random.seed(seed)
                    y_obs=np.zeros((steps,))
                    y_obs[0]=y_ini

                    AMOC.Runge_Kutta_dynamics(
                        F,y_obs,mu,steps,dt)

                    Q_obs=AMOC.salinity_flux_to_flow_strength(
                        mu,y_obs,td,ita,V)
                    Q_obs_nonnoise=Q_obs.copy()
                    Q_obs+=np.random.randn(steps)*s_obs
                    ##start data assimilation 
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

                    p_num=7

                    st_inds=[ita_ind,V_ind,mu_ind,td_ind]
                    
                    obs_steps=[int(gap/dt)+(t+1)*fs for t in range(obs_num)]
                
                    pcls.random_sampling_amoc()

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
                    Q_results[:,0]=AMOC.salinity_flux_to_flow_strength(
                        pcls.particle[:,mu_ind],
                        y_results[:,0],
                        pcls.particle[:,td_ind],
                        pcls.particle[:,ita_ind],
                        pcls.particle[:,V_ind]
                    )

                    y2_results[:,0]=y_ini ##set initial condition
                    Q2_results[:,0]=AMOC.salinity_flux_to_flow_strength(
                        pcls.particle[:,mu_ind],
                        y2_results[:,0],
                        pcls.particle[:,td_ind],
                        pcls.particle[:,ita_ind],
                        pcls.particle[:,V_ind]
                    )

                    y3_results[:,0]=y_ini ##set initial condition
                    Q3_results[:,0]=AMOC.salinity_flux_to_flow_strength(
                        pcls.particle[:,mu_ind],
                        y3_results[:,0],
                        pcls.particle[:,td_ind],
                        pcls.particle[:,ita_ind],
                        pcls.particle[:,V_ind]
                    )

                    pcls.particle[:,y_ind]=y_ini
                    pcls.particle[:,Q_ind]=AMOC.salinity_flux_to_flow_strength(
                        pcls.particle[:,mu_ind],
                        pcls.particle[:,y_ind],
                        pcls.particle[:,td_ind],
                        pcls.particle[:,ita_ind],
                        pcls.particle[:,V_ind]        
                    )

                    pcls.particle[:,y2_ind]=y_ini
                    pcls.particle[:,Q2_ind]=AMOC.salinity_flux_to_flow_strength(
                        pcls.particle[:,mu_ind],
                        pcls.particle[:,y2_ind],
                        pcls.particle[:,td_ind],
                        pcls.particle[:,ita_ind],
                        pcls.particle[:,V_ind]        
                    )
                    pcls.particle[:,y3_ind]=y_ini
                    pcls.particle[:,Q3_ind]=AMOC.salinity_flux_to_flow_strength(
                        pcls.particle[:,mu_ind],
                        pcls.particle[:,y3_ind],
                        pcls.particle[:,td_ind],
                        pcls.particle[:,ita_ind],
                        pcls.particle[:,V_ind]        
                    )

                    for step in range(steps):
                        pcls.particle[:,pre_y_ind]=pcls.particle[:,y_ind]
                        pcls.particle[:,y_ind]=AMOC.simple_model(
                            pcls.particle[:,pre_y_ind],
                            step,F,
                            pcls.particle[:,mu_ind],
                            dt
                        )
                        pcls.particle[:,Q_ind]=AMOC.salinity_flux_to_flow_strength(
                            pcls.particle[:,mu_ind],
                            pcls.particle[:,y_ind],
                            pcls.particle[:,td_ind],
                            pcls.particle[:,ita_ind],
                            pcls.particle[:,V_ind]
                        )
                        
                        pcls.particle[:,pre_y2_ind]=pcls.particle[:,y2_ind]
                        pcls.particle[:,y2_ind]=AMOC.simple_model(
                            pcls.particle[:,pre_y2_ind],
                            step,F2,
                            pcls.particle[:,mu_ind],
                            dt
                        )
                        pcls.particle[:,Q2_ind]=AMOC.salinity_flux_to_flow_strength(
                            pcls.particle[:,mu_ind],
                            pcls.particle[:,y2_ind],
                            pcls.particle[:,td_ind],
                            pcls.particle[:,ita_ind],
                            pcls.particle[:,V_ind]
                        )

                        pcls.particle[:,pre_y3_ind]=pcls.particle[:,y3_ind]
                        pcls.particle[:,y3_ind]=AMOC.simple_model(
                            pcls.particle[:,pre_y3_ind],
                            step,F3,
                            pcls.particle[:,mu_ind],
                            dt
                        )
                        pcls.particle[:,Q3_ind]=AMOC.salinity_flux_to_flow_strength(
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
                            inds=pcls.resampling(weights)
                            pcls.particle=pcls.particle[inds,:]

                            pcls.gaussian_inflation(st_inds,a=0.1)
                        
                        pcls.particle[:,tip_step_ind]+=((pcls.particle[:,y_ind]>yth)-pcls.particle[:,tip_ind])*step*(1-pcls.particle[:,tip_ind])
                        pcls.particle[:,tip_ind]=(pcls.particle[:,tip_step_ind]>0)

                        pcls.particle[:,tip_step2_ind]+=((pcls.particle[:,y2_ind]>yth)-pcls.particle[:,tip2_ind])*step*(1-pcls.particle[:,tip2_ind])
                        pcls.particle[:,tip2_ind]=(pcls.particle[:,tip_step2_ind]>0)

                        pcls.particle[:,tip_step3_ind]+=((pcls.particle[:,y3_ind]>yth)-pcls.particle[:,tip3_ind])*step*(1-pcls.particle[:,tip3_ind])
                        pcls.particle[:,tip3_ind]=(pcls.particle[:,tip_step3_ind]>0)

                        y_results[:,step]+=pcls.particle[:,y_ind]
                        Q_results[:,step]+=pcls.particle[:,Q_ind]

                        y2_results[:,step]+=pcls.particle[:,y2_ind]
                        Q2_results[:,step]+=pcls.particle[:,Q2_ind]

                        y3_results[:,step]+=pcls.particle[:,y3_ind]
                        Q3_results[:,step]+=pcls.particle[:,Q3_ind]

                        mu_results[:,step]+=pcls.particle[:,mu_ind]
                        V_results[:,step]+=pcls.particle[:,V_ind]
                        ita_results[:,step]+=pcls.particle[:,ita_ind]

                    num=len(pcls.particle[pcls.particle[:,y_ind]>yth])
                    f.write('{},{},{},{}\n'.format(1,r_obs,obs_num,num))

                    num2=len(pcls.particle[pcls.particle[:,y2_ind]>yth])
                    f.write('{},{},{},{}\n'.format(2,r_obs,obs_num,num2))

                    num3=len(pcls.particle[pcls.particle[:,y3_ind]>yth])
                    f.write('{},{},{},{}\n'.format(3,r_obs,obs_num,num3))
        output='../../output/final_result/amoc/scenario_freq_amp4'
        shutil.move('./output.log',output)