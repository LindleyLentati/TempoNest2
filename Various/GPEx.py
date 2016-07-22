import numpy as np
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
from scipy.optimize import minimize
import scipy as sp


N_data = 10



Margindata_x=np.arange(1,N_data+1)

MarginCovD = np.zeros([N_data,N_data])
for i in range(N_data):
	for j in range(N_data):
		MarginCovD[i][j] = -0.5*(Margindata_x[i] - Margindata_x[j])**2


MarginCovD2 = np.zeros([N_data,N_data])
for i in range(N_data):
	for j in range(N_data):
		MarginCovD2[i][j] = (Margindata_x[i] - Margindata_x[j])


sig_f = 5
sig_n = 1
l = 10

Cov_Matern = sig_f**2*(1+np.sqrt(3.0)*MarginCovD2/l)*np.exp(-np.sqrt(3.0)*MarginCovD2/l) 

Cov_M = sig_f**2*np.exp(np.copy(MarginCovD)/l**2) 
Cov_N = sig_n**2*np.ones(N_data)

Sigma = Cov_M + np.diag(Cov_N)
c = sp.linalg.cholesky(Sigma, lower=True)


ran=np.random.normal(0,1,N_data)
data=np.dot(c, ran)

plt.plot(Margindata_x, data)
plt.show()


n_params=N_data+3

x0 = np.array(np.zeros(n_params))

x0[:N_data] = data
x0[N_data+0] = 5
x0[N_data+1] = 1
x0[N_data+2] = 10

def like(x):

	amps = x[:N_data]
	sig_f = x[N_data+0]
	sig_n = x[N_data+1]
	l = x[N_data+2]


	Cov_M = sig_f**2*np.exp(np.copy(CovD)/l**2) + np.diag(np.ones(N_data)*sig_n**2)
	Cov_N = np.ones(N_data)

	datalike = -0.5*np.sum((data - amps)**2/Cov_N**2)
	detN = -0.5*np.sum(np.log(Cov_N))
	detC = -0.5*np.linalg.slogdet(Cov_M)[1]


	try:	
		priorlike = -0.5*np.dot(np.linalg.lstsq(Cov_M, amps)[0], amps)
	except:
		return -np.inf


	like = datalike + priorlike + detN + detC

	return like


def Scaledlike(x):

	amps = x[:N_data]
	sig_f = x[N_data+0]
	sig_n = 1.0#x[N_data+1]
	l = x[N_data+2]

	#print sig_f, sig_n, l


	Cov_M = sig_f**2*np.exp(np.copy(CovD)/l**2) 
	Cov_N = sig_n**2*np.ones(N_data)

	try:
		Chol_M = sp.linalg.cholesky(Cov_M)
	except:
		return -np.inf


	TransAmps=np.dot(Chol_M, amps)

	datalike = -0.5*np.sum((data - TransAmps)**2/Cov_N**2)
	detN = -0.5*np.sum(np.log(Cov_N))
	#detC = -0.5*np.linalg.slogdet(Cov_M)[1]

	priorlike = -0.5*np.dot(amps, amps)

	like = datalike + priorlike + detN #+ detC

	if(doplot == True):
		plt.plot(data_x, amps)
		plt.plot(data_x, data)
		plt.show()

	return like


def my_prior(x):
    logp = 0.

    if np.all(x <= pmax) and np.all(x >= pmin):
	logp = np.sum(np.log(1/(pmax-pmin)))
    else:
	logp = -np.inf

    return logp

parameters = []
for i in range(N_data):
	parameters.append('A'+str(i))
parameters.append('sig_f')
parameters.append('sig_n')
parameters.append('l')


print parameters
n_params = len(parameters)
print n_params

    
pmin = np.array(np.ones(n_params))*-10
pmax = np.array(np.ones(n_params))*10


pmin[N_data+0] = 0
pmin[N_data+1] = 0
pmin[N_data+2] = 0

pmax[N_data+0] = 20
pmax[N_data+1] = 20
pmax[N_data+2] = 20


x0 = np.array(np.zeros(n_params))

x0[N_data+0] = 1
x0[N_data+1] = 1
x0[N_data+2] = 1




cov_diag = np.array(np.ones(n_params))


burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=like,logp=my_prior,
                            cov=np.diag(cov_diag**2), outDir='./Chains/',resume=False)
#sampler.addProposalToCycle(lfunc.TimeJump, 20)
sampler.sample(p0=x0,Niter=100000,isave=10,burn=burnin,thin=1,neff=10000)

chains=np.loadtxt('./chains/chain_1.txt').T
ML=chains.T[np.argmax(chains[-3][burnin:])][:n_params]
doplot = True
like(ML)





N_data = 100



Margindata_x=np.arange(1,101)

MarginCovD = np.zeros([N_data,N_data])
for i in range(N_data):
	for j in range(N_data):
		MarginCovD[i][j] = -0.5*(Margindata_x[i] - Margindata_x[j])**2

sig_f = 5
sig_n = 1
l = 10

Cov_M = sig_f**2*np.exp(np.copy(MarginCovD)/l**2) 
Cov_N = sig_n**2*np.ones(N_data)

Sigma = Cov_M + np.diag(Cov_N)
c = sp.linalg.cholesky(Sigma, lower=True)


ran=np.random.normal(0,1,N_data)
data=np.dot(c, ran)

def Marginlike(x):

	sig_f = x[0]
	sig_n = 1.0#x[N_data+1]
	l = x[2]

	#print sig_f, sig_n, l


	Cov_M = sig_f**2*np.exp(np.copy(MarginCovD)/l**2) 
	Cov_N = sig_n**2*np.ones(N_data)

	Sigma = Cov_M + np.diag(Cov_N)
	
	try:
		detC = -0.5*np.linalg.slogdet(Sigma)[1]
		Chol_S = sp.linalg.cho_factor(Sigma)	
		Mlike = -0.5*np.dot(sp.linalg.cho_solve(Chol_S, data), data)


	except:
		return -np.inf
	


	like = detC+Mlike


	return like

plt.plot(Margindata_x, data)
plt.plot(Margindata_x,)



parameters = []
parameters.append('sig_f')
parameters.append('sig_n')
parameters.append('l')


print parameters
n_params = len(parameters)
print n_params

    
pmin = np.array(np.ones(n_params))
pmax = np.array(np.ones(n_params))


pmin[0] = 0
pmin[1] = 0
pmin[2] = 0

pmax[0] = 20
pmax[1] = 20
pmax[2] = 20


x0 = np.array(np.zeros(n_params))

x0[0] = 1
x0[1] = 1
x0[2] = 1




cov_diag = np.array(np.ones(n_params))


burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=Marginlike,logp=my_prior,
                            cov=np.diag(cov_diag**2), outDir='./MarginChains/',resume=False)
#sampler.addProposalToCycle(lfunc.TimeJump, 20)
sampler.sample(p0=x0,Niter=100000,isave=10,burn=burnin,thin=1,neff=10000)

chains=np.loadtxt('./Chains/chain_1.txt').T
ML=chains.T[np.argmax(chains[-3][burnin:])][:n_params]
doplot = True
like(ML)

