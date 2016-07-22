import numpy as np
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
from scipy.optimize import minimize
import scipy as sp

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

plt.plot(Margindata_x, data)
plt.show()

def my_prior(x):
    logp = 0.

    if np.all(x <= pmax) and np.all(x >= pmin):
	logp = np.sum(np.log(1/(pmax-pmin)))
    else:
	logp = -np.inf

    return logp

def Marginlike(x):

	sig_f = x[0]
	sig_n = x[1]
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

x0[0] = 5
x0[1] = 1
x0[2] = 10




cov_diag = np.array(np.ones(n_params))


burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=Marginlike,logp=my_prior,
                            cov=np.diag(cov_diag**2), outDir='./MarginChains/',resume=False)
#sampler.addProposalToCycle(lfunc.TimeJump, 20)
sampler.sample(p0=x0,Niter=10000,isave=10,burn=burnin,thin=1,neff=10000)

chains=np.loadtxt('./MarginChains/chain_1.txt').T
plt.close()
for i in range(3):
	plt.scatter(chains[i][100:], chains[-4][100:])
	plt.show()



