###状態空間モデルの実装##################
##パーティクルフィルタを用いて実装する
##観測に付するノイズは多変数関数にしておいた方がいい？
#####################################
import pandas as pd
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import subprocess

class ParticleFilter(object):
  ##set params which are the constant
  def __init__(self,n_particle):
    self.n_particle = n_particle #粒子数

    #TODO:set the covariance matrix of observation noise(arbitral value)

    
  def random_sampling(self):#n_particleの5次元粒子を生成（Have to include the state variables as particle)
    g = rd.uniform(0,2.0,self.n_particle)
    v=np.zeros((self.n_particle))
    Tl=np.zeros((self.n_particle,))
    pre_v=np.zeros((self.n_particle))
    pre_Tl=np.zeros((self.n_particle,))
    #alpha = rd.uniform(4.5,5.5,self.n_particle)
    #beta = rd.uniform(9.5,10.5,self.n_particle)
    #gamma = rd.uniform(0.15,0.25,self.n_particle)
    #Tf = rd.uniform(30,35,self.n_particle) #TODO: 仮置き 大まかな値を予想して生成する必要あり
    self.particle = np.stack([g,v,Tl,pre_v,pre_Tl]).T
    
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

  def hist_particle(self, dir, num = 0):
    for i in range(5):
      plt.hist(self.particle[:, i],bins = 20)
      plt.savefig(f"{dir}/{i}_{num}.png")
      plt.clf()
    return

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
  
  #get observation and resample param
  #input observation should be ndarray
  def renew_param(self,obs,ind): #ind:index list of state variable(input observation)
    ##make likelifood matrix
    s=1 #TODO:set the uncertainty of the variable

    weights_mat=self.log_norm_likelihood(obs,self.particle[:,ind],s)
    weights=np.sum(weights_mat,axis=0)

    self.resampling(np.exp(weights))
    
    return #then, renew self.particle[:,ind] by time evolution according to the governing equation 
  