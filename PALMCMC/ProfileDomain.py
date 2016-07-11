import numpy as np
import psrchive
from libstempo.libstempo import *
import libstempo as T
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
from scipy.optimize import minimize
from Class import *


lfunc = Likelihood()

lfunc.loadPulsar("OneChan.par", "OneChan.tim")




'''Get initial Fit to the Profile'''

lfunc.TScrunch(doplot = True)

lfunc.getInitialParams(MaxCoeff = 10)


'''Make interpolation Matrix'''

lfunc.PreComputeShapelets(interpTime = 1, MeanBeta = lfunc.MeanBeta)

lfunc.getInitialPhase()



parameters = []
parameters.append('Phase')
parameters.append('NCoeff')
for i in range(lfunc.MaxCoeff-1):
	parameters.append('S'+str(i))
for i in range(lfunc.numTime):
	parameters.append(lfunc.psr.pars()[i])



print parameters
n_params = len(parameters)
print n_params

    
pmin = np.array(np.ones(n_params))*-1024
pmax = np.array(np.ones(n_params))*1024


pmin[1] = 1
pmax[1] = lfunc.MaxCoeff  #Ncoeff
pmin[2:2+lfunc.MaxCoeff-1] = -1
pmax[2:2+lfunc.MaxCoeff-1] = 1

lfunc.pmin = pmin
lfunc.pmax = pmax

x0 = np.array(np.zeros(n_params))

pcount = 0
x0[pcount] = lfunc.MeanPhase
pcount += 1

x0[pcount] = 9
pcount += 1


for i in range(lfunc.MaxCoeff-1):
	x0[pcount+i] = lfunc.MLShapeCoeff[1+i]

#x0[pcount:pcount+len(ShapeMax)] = ShapeMax
pcount += lfunc.MaxCoeff-1

for i in range(lfunc.numTime):
	x0[pcount+i] = 0
pcount += lfunc.numTime




cov_diag = np.array(np.ones(n_params))

pcount = 0
cov_diag[pcount] = 0.0000115669002113
pcount += 1
cov_diag[pcount] = 1
pcount += 1
for i in range(lfunc.MaxCoeff-1):
        cov_diag[pcount+i] = lfunc.hess[i]
pcount += lfunc.MaxCoeff-1
for i in range(lfunc.numTime):
        cov_diag[pcount+i] = 1
pcount += lfunc.numTime




lfunc.doplot=False
burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=lfunc.MarginLogLike,logp=lfunc.my_prior,
                            cov=np.diag(cov_diag**2), outDir='./chains/',resume=False)
#sampler.addProposalToCycle(lfunc.TimeJump, 20)
sampler.sample(p0=x0,Niter=10000,isave=10,burn=burnin,thin=1,neff=1000)

'''
chains=np.loadtxt('./chains/chain_1.txt').T
ML=chains.T[np.argmax(chains[-3][burnin:])][:n_params]
ML[0]=3.36203222e-03
x0=ML
doplot=True
MarginLogLike(x0)
doplot=False

np.savetxt("ML.dat", ML)

chains=np.loadtxt('./chains/chain_1.txt').T
ML=chains.T[np.argmax(chains[-3][burnin:])][:n_params]
STD=np.zeros(n_params)
for i in range(n_params):
	STD[i]  =  np.std(chains[i][burnin:])
	print "param:", i, np.mean(chains[i][burnin:]), np.std(chains[i][burnin:])
np.savetxt("Cov.dat", STD)
cov_diag = STD
x0=ML
'''

