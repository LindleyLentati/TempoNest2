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
lfunc.loadPulsar("OneChan.par", "OneChan.tim", root='Sim1-OneChan')


'''Get initial Fit to the Profile'''

lfunc.TScrunch(doplot = True, channels = 1)

lfunc.getInitialParams(MaxCoeff = 20, cov_diag=[0.01, 0.1, 0.1], resume=False, outDir = './InitFFTMNChains/Max20-', sampler='multinest', incScattering = False, mn_live = 1000,  fitNComps = 1, doplot = True)


'''Make interpolation Matrix'''


lfunc.PreComputeFFTShapelets(interpTime = 1, MeanBeta = lfunc.MeanBeta, doplot=True)

lfunc.getInitialPhase(doplot = True)


lfunc.ScatterInfo = lfunc.GetScatteringParams(mode = 'parfile')

parameters = []
parameters.append('Phase')
for i in range(lfunc.TotCoeff-1):
	for j in range(lfunc.EvoNPoly+1):
		parameters.append('S'+str(i+1)+'E'+str(j))
for i in range(lfunc.numTime):
	parameters.append(lfunc.psr.pars()[i])
for i in range(lfunc.NScatterEpochs):
	parameters.append("Scatter_"+str(i))


print parameters
n_params = len(parameters)
print n_params
lfunc.n_params = n_params
    
pmin = np.array(np.ones(n_params))*-100
pmax = np.array(np.ones(n_params))*100

for i in range(lfunc.NScatterEpochs):
	pmin[-lfunc.NScatterEpochs+i] = -6
	pmax[-lfunc.NScatterEpochs+i] = 1

lfunc.pmin = pmin
lfunc.pmax = pmax

x0 = np.array(np.zeros(n_params))

pcount = 0
x0[pcount] = lfunc.MeanPhase
pcount += 1

for i in range(lfunc.TotCoeff-1):
	for j in range(lfunc.EvoNPoly+1):
		x0[pcount] = lfunc.MLShapeCoeff[1+i][j]
		pcount += 1


for i in range(lfunc.numTime):
	x0[pcount+i] = 0
pcount += lfunc.numTime
for i in range(lfunc.NScatterEpochs):
	x0[pcount+i] = lfunc.MeanScatter
pcount += lfunc.NScatterEpochs


lfunc.calculateFFTHessian(x0)
covM=np.linalg.inv(lfunc.hess)
lfunc.PhasePrior = np.sqrt(covM[0,0])*lfunc.ReferencePeriod
lfunc.MeanPhase = x0[0]*lfunc.ReferencePeriod



lfunc.doplot=False
burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=lfunc.FFTMarginLogLike,logp=lfunc.my_prior,
                            cov=covM, outDir='./Chains/',resume=False)
sampler.sample(p0=x0,Niter=20000,isave=10,burn=burnin,thin=1,neff=1000)

chains=np.loadtxt('./Chains/chain_1.txt').T


import corner as corner
chains=chains[:,burnin:]
if(lfunc.numTime > 0):
	Tchains = chains[1+lfunc.TotCoeff-1:1+lfunc.TotCoeff-1 + lfunc.numTime]
	figure = corner.corner(Tchains.T, labels=[r"$RA$", r"$DEC$", r"$F0$", r"$F1$"],
		               quantiles=[0.16, 0.5, 0.84],
		               show_titles=True, title_kwargs={"fontsize": 12})

ML=chains.T[np.argmax(chains[-3])][:n_params]
lfunc.WaterFallPlot(ML)


