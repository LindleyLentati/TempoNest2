import numpy as np
from Class import *

lfunc = Likelihood(useGPU = True)
lfunc.loadPulsar("OneChan.par", "OneChan.tim", root='./results/GHS-OneChan-')


'''Get initial Fit to the Profile'''

lfunc.TScrunch(doplot = False, channels = 1)

lfunc.getInitialParams(MaxCoeff = 20, cov_diag=[0.01, 0.1, 0.1], resume=True, outDir = './InitFFTMNChains/Max20-', sampler='multinest', incScattering = False, mn_live = 1000,  fitNComps = 1, doplot = False)


'''Make interpolation Matrix'''

lfunc.PreComputeFFTShapelets(interpTime = 1, MeanBeta = lfunc.MeanBeta, doplot=False)

lfunc.getInitialPhase(doplot = False)

lfunc.ScatterInfo = lfunc.GetScatteringParams(mode = 'parfile')

parameters = []
for i in range(lfunc.NToAs):
	parameters.append('ProfileAmp_'+str(i))
for i in range(lfunc.NToAs):
	parameters.append('ProfileNoise_'+str(i))
	
parameters.append('Phase')
for i in range(lfunc.numTime):
	parameters.append(lfunc.psr.pars()[i])
	
for i in range(lfunc.TotCoeff-1):
	for j in range(lfunc.EvoNPoly+1):
		parameters.append('S'+str(i+1)+'E'+str(j))

n_params = len(parameters)
print(n_params)
lfunc.n_params = n_params


lfunc.PhasePrior = 1e-5
lfunc.startPoint, lfunc.EigV, lfunc.EigM, lfunc.hess = lfunc.calculateGHSHessian(diagonalGHS=False)

DenseParams = lfunc.startPoint[2*lfunc.NToAs:]
PrincipleParams = np.dot(lfunc.EigM.T, DenseParams)
lfunc.startPoint[2*lfunc.NToAs:] = PrincipleParams

lfunc.callGHS(resume=False, nburn = 100, nsamp = 5000, feedback_int = 100, seed = -1, max_steps = 10, dim_scale_fact = 0.4)
