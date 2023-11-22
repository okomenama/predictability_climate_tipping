###状態空間モデルの実装##################
##パーティクルフィルタを用いて実装する
##観測に付するノイズは多変数関数にしておいた方がいい？
#####################################
import pandas as pd
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess

class ParticleFilter(object):
  ##set params which are the constant
  def __init__(self,n_particle):
    self.n_particle = n_particle #粒子数

    #TODO:set the covariance matrix of observation noise(arbitral value)

    
  def random_sampling(self):#n_particleの5次元粒子を生成（Have to include the state variables as particle)
    Topt = rd.uniform(26,34.0,self.n_particle)
    #Topt=np.ones((self.n_particle,))*28
    g0= rd.uniform(0,2.5,self.n_particle)
    #gamma = rd.uniform(0.1,0.3,self.n_particle)
    gamma=np.ones((self.n_particle,))*0.2
    alpha=np.ones((self.n_particle,))*5
    beta=np.ones((self.n_particle,))*10
    #alpha = rd.uniform(3.5,6.5,self.n_particle)
    #beta = rd.uniform(8.0,12.0,self.n_particle)

    v=np.zeros((self.n_particle))
    Tl=np.zeros((self.n_particle,))
    g=np.zeros((self.n_particle,))
    pre_v=np.zeros((self.n_particle))
    pre_Tl=np.zeros((self.n_particle,))
    pre_g=np.zeros((self.n_particle,))
    tip=np.zeros((self.n_particle,))
    tip_step=np.zeros((self.n_particle,))

    v2=np.zeros((self.n_particle))
    Tl2=np.zeros((self.n_particle,))
    g2=np.zeros((self.n_particle,))
    pre_v2=np.zeros((self.n_particle))
    pre_Tl2=np.zeros((self.n_particle,))
    pre_g2=np.zeros((self.n_particle,))
    tip2=np.zeros((self.n_particle,))
    tip2_step=np.zeros((self.n_particle,))

    v3=np.zeros((self.n_particle))
    Tl3=np.zeros((self.n_particle,))
    g3=np.zeros((self.n_particle,))
    pre_v3=np.zeros((self.n_particle))
    pre_Tl3=np.zeros((self.n_particle,))
    pre_g3=np.zeros((self.n_particle,))
    tip3=np.zeros((self.n_particle,))
    tip3_step=np.zeros((self.n_particle,))
    #Tf = rd.uniform(30,35,self.n_particle) #TODO: 仮置き 大まかな値を予想して生成する必要あり
    self.particle = np.stack([
      Topt,g0,gamma,alpha,beta,
      v,Tl,g,pre_v,pre_Tl,pre_g,tip,tip_step,
      v2,Tl2,g2,pre_v2,pre_Tl2,pre_g2,
      v3,Tl3,g3,pre_v3,pre_Tl3,pre_g3,
      tip2,tip2_step,tip3,tip3_step]).T

  def random_sampling_amoc(self):#generate random particles of AMOC model
    ita=np.random.uniform(10**(-5),5*10**(-5),self.n_particle)
    V=np.ones((self.n_particle,))*300*4.5*8250
    mu=np.ones((self.n_particle,))*6.2**(0.5)
    td=np.ones((self.n_particle,))*180

    y=np.zeros((self.n_particle,))
    pre_y=np.zeros((self.n_particle,))
    Q=np.zeros((self.n_particle,))

    self.particle = np.stack([ita,V,mu,td,y,pre_y,Q]).T


  ####still for two-dimensional variables
  def gaussian_inflation(self,st_inds,a=0.1):
    cov=np.cov(self.particle[:,st_inds].transpose())
    #print('cov=')
    #print(cov)
    self.particle[:,st_inds]+=a*np.random.multivariate_normal(np.zeros(len(st_inds)),cov,self.n_particle)
    return
    
  def norm_likelihood(self,y,x,s):
    return ((np.sqrt(2*np.pi))*s)**(-1)*np.exp(-(y-x)**2/(2*s**2))
  
  def log_norm_likelihood(self,y,x,s):
    return -np.log(s)-(y-x)**2/(2*s**2)-np.log(2*np.pi)/2
  
  def multi_dim_norm_likelihood(self,v_sys,Tl_sys,v_obs,Tl_obs,R):
    obs=np.array([v_obs,Tl_obs]).reshape((2,1))
    sys=np.array([v_sys,Tl_sys]).reshape((2,1))
    dif=obs-sys

    detR=np.linalg.det(R)  ##R:observation covariance matrix detR has to be other than 0
    return -np.log(2*np.pi*detR)/2-(np.dot(np.dot(dif.T,np.linalg.inv(R)),dif))/2 ##check this is scaler value or not
    
  ##尤度を与えるとそれ以下のところに何個の粒子があるかを返す関数
  def F_inv(self,w_cumsum,u):
    # w_cumsumが単調増加であることを仮定
    return np.searchsorted(w_cumsum, u)
  ##indexの配列で返す
  def resampling(self,weights):
    weights /= weights.sum()
    w_cumsum = np.cumsum(weights)
    k_list = np.searchsorted(w_cumsum, rd.uniform(0,1,size = self.n_particle))
    return k_list
  
  def resampling2(self,weights):
    u0 = rd.uniform(0,1/self.n_particle)
    u = [1/self.n_particle*i +u0 for i in range(self.n_particle)]
    w_cumsum = np.cumsum(weights)
    k = np.array([self.F_inv(w_cumsum,val) for val in u])
    return k

  def hist_particle(self, dir, num):
    for i in range(num):
      plt.hist(self.particle[:, i],bins = 20)
      plt.savefig(f"{dir}/{i}+particle.png")
      plt.clf()
    return
  
  def write_particle_dist(self,ind1,ind2,dir, num=0):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x=self.particle[:,ind1]
    y=self.particle[:,ind2]
    H = ax.hist2d(x,y, bins=40, cmap=cm.jet)
    ax.set_title('1st graph')

    fig.colorbar(H[3],ax=ax)
    fig.savefig(f"{dir}/heatmap.png")
    fig.clf()

  def resampling3(self,weights):
    stdparticle = np.std(self.particle, axis=0)
    weights /= weights.sum()
    wmed = np.median(weights)
    weights[weights < wmed] = 0
    max_index = np.argmax(weights)
    k = self.resampling(weights)
    self.particle = self.particle[k]
    
    for i in range(self.n_particle):
      if (weights[i] == 0):
        self.particle[i] = self.particle[max_index]+ rd.normal(0., stdparticle) / 2.

    return
    
