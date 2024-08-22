#import os
#os.chdir('/home/andrea/Desktop/FinNetworks/')
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
n_cores = int(multiprocessing.cpu_count()*0.9)
import matplotlib.pyplot as plt


d = 30
dt = 2e-4
#T = 0.1
n_replicas = int(1e7)

sigma = 100.
#gamma = 3.
beta = 0.01

sqrt_dt = np.sqrt(dt)
sqrt_beta = np.sqrt(beta)


len_gamma_list = 10
gamma_list = [0.5*np.power(1.5,i) for i in range(len_gamma_list)]
T_list = [0.1, 0.5]

def dynamics(gamma,T):
        M = np.random.normal(0,1,(d,d))
        #for i in range(d): 
            #M[i,i]=0           
        t = 0
        x = np.array([0.]*d)
        h = np.array([0.]*d)
        while t < T:
            t += dt
            dW = np.random.normal(0,sqrt_dt,d)
            pre_x = x.copy()
            x += sigma*dW + gamma*M.dot(h)*dt 
            h += sqrt_beta*(x-pre_x) -beta*h*dt
        return np.var(x, ddof=1)

def EEy(gamma,T):         
    realizations = Parallel(n_jobs=n_cores)(delayed(dynamics)(gamma,T) for j in range(n_replicas))      
    EEy_gamma = [gamma,np.mean(realizations),np.sqrt(np.var(realizations, ddof=1))]     
    return EEy_gamma

EEy_d_gamma = []
for T in T_list:
    this_T = []
    for gamma in gamma_list:
        print('gamma = ' + str(gamma))
        this = EEy(gamma,T)
        this_T.append(this)
    EEy_d_gamma.append(this_T)

sigma_2 = np.power(sigma,2)
T = T_list[0]
factor = sigma_2*(beta/3)*(d+1)*np.power(T,3)
T = T_list[1]
factor2 = sigma_2*(beta/3)*(d+1)*np.power(T,3)


fig, ax = plt.subplots()
plt.xlabel('$\gamma$', fontsize=14)
plt.ylabel('$\mathbb{E}y -\sigma^2 t$', fontsize=14)
T = T_list[0]
ax.scatter(gamma_list, [k[1]-sigma_2*T for k in EEy_d_gamma[0]], color="black")
theory = [factor*np.power(g,2) for g in gamma_list]
plt.plot(gamma_list,theory,linestyle ='dashed', color="black")
T = T_list[1]
ax.scatter(gamma_list, [k[1]-sigma_2*T for k in EEy_d_gamma[1]], color="gray")
theory2 = [factor2*np.power(g,2) for g in gamma_list]
plt.plot(gamma_list,theory2,linestyle ='dashed', color="gray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(['t = 0.1', 'Theory', 't = 0.5', 'Theory'])
plt.savefig('Fig2.pdf', bbox_inches='tight')


with open(r'results.txt', 'w') as fp:
    fp.write("\n".join(str(item) for item in EEy_d_gamma[0]))
    fp.write("\n".join(str(item) for item in EEy_d_gamma[1]))


# THEORY
#gamma_2 = np.power(gamma,2) 
#first = sigma_2*T
#second = first + (sigma_2*gamma*sqrt_beta/(d-1))*(M.trace()-np.sum(M)/d)*np.power(T,2)
#M_tilde = M + M.transpose()
#third = second + ( -(beta/3)*(sigma_2*gamma*sqrt_beta/(d-1))*(M.trace()-np.sum(M)/d) + (sigma_2*gamma_2*beta/(3*(d-1))) * np.sum(np.multiply(M,(M_tilde - np.array([np.sum(M_tilde,axis=1)/d]*d)))) ) * np.power(T,3)

