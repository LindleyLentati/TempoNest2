from __future__ import absolute_import, unicode_literals, print_function
from ctypes import cdll
import sys

from ctypes import *
from numpy.ctypeslib import as_array
import signal
import inspect
import ctypes

import numpy as np
import scipy.linalg as sl
import psrchive
from libstempo.libstempo import *
import libstempo as T
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
from scipy.optimize import minimize
from Class import *
import copy


libname = 'libghs'

libname += {
	'darwin' : '.dylib',
	'win32'  : '.dll',
	'cygwin' : '.dll',
}.get(sys.platform, '.so')


lib = cdll.LoadLibrary(libname)



def interrupt_handler(signal, frame):
	sys.stderr.write('ERROR: Interrupt received: Terminating\n')
	sys.exit(1)
	
	
c_double_p=ctypes.POINTER(ctypes.c_double)



def run_guided_hmc( neg_logpost,
        			write_extract,
        			num_dim, 
					start_point,
        			step_sizes,
        			file_prefix,
        			dim_scale_fact = 0.4,
        			max_steps = 10,
        			seed = -1,
        			resume = 1,
        			feedback_int = 100,
        			nburn = 1000,
        			nsamp = 10000,
        			doMaxLike = 0):

	"""
	Runs the GHS
		
	"""

	loglike_type = CFUNCTYPE(None, POINTER(c_int), POINTER(c_double),
		POINTER(c_double), POINTER(c_double))

	write_extract_type  = CFUNCTYPE(None, POINTER(c_int), POINTER(c_double),
		POINTER(c_double), POINTER(c_double))
		
	c_double_p=ctypes.POINTER(ctypes.c_double)
		
	lib.run_guided_hmc( c_int(num_dim),
							start_point.ctypes.data_as(c_double_p),
							c_double(dim_scale_fact),
							c_int(max_steps),
							step_sizes.ctypes.data_as(c_double_p),
							create_string_buffer(file_prefix.encode(),100),
							c_int(seed),
							c_int(resume),
							c_int(feedback_int),
							loglike_type(neg_logpost),
							write_extract_type(write_extract),
							c_int(nburn),
							c_int(nsamp),
							c_int(doMaxLike))






def write_gauss_ghs_extract_with_logpostval(ndim, x, logpostval, grad):

	params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))
	
	
	#Send relevant parameters to physical coordinates for likelihood
	
	DenseParams = params[2*lfunc.NToAs:]
	PhysParams = np.dot(EigM, DenseParams)
	params[2*lfunc.NToAs:] = PhysParams

	for i in range(ndim[0]):
		test_gauss_outfile.write(str(params[i])+" ")
        
	test_gauss_outfile.write(str(logpostval[0])+"\n")      
	
	return






lfunc = Likelihood()
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
	
print(parameters)
n_params = len(parameters)
print(n_params)
lfunc.n_params = n_params



def calculateGHSHessian(diagonalGHS = False):

	NCoeff = lfunc.TotCoeff-1

	x0 = np.zeros(lfunc.n_params)
	cov_diag = np.zeros(lfunc.n_params)
	
	DenseParams = 1  + lfunc.numTime + NCoeff*(lfunc.EvoNPoly+1) 
	
	hess_dense = np.zeros([DenseParams,DenseParams])

	

	xS = lfunc.ShiftedBinTimes[:,0]-lfunc.MeanPhase*lfunc.ReferencePeriod
	xS = ( xS + lfunc.ReferencePeriod/2) % (lfunc.ReferencePeriod ) - lfunc.ReferencePeriod/2

	InterpBins = (xS%(lfunc.ReferencePeriod/lfunc.Nbins[:])/lfunc.InterpolatedTime).astype(int)
	WBTs = xS-lfunc.InterpolatedTime*InterpBins
	RollBins=(np.round(WBTs/(lfunc.ReferencePeriod/lfunc.Nbins[:]))).astype(np.int)

	ShapeAmps=lfunc.MLShapeCoeff

	s = np.sum([np.dot(InterpFBasis2[InterpBins], ShapeAmps[:,i])*(((lfunc.psr.freqs - lfunc.EvoRefFreq)/1000.0)**i).reshape(lfunc.NToAs,1) for i in range(lfunc.EvoNPoly+1)], axis=0)
	
	j = np.sum([np.dot(InterpJBasis2[InterpBins], ShapeAmps[:,i])*(((lfunc.psr.freqs - lfunc.EvoRefFreq)/1000.0)**i).reshape(lfunc.NToAs,1) for i in range(lfunc.EvoNPoly+1)], axis=0)


	for i in range(lfunc.NToAs):


		rfftfreqs=np.linspace(1,lfunc.NFBasis,lfunc.NFBasis)/lfunc.Nbins[i]

		RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
		ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

	
		RollData = np.zeros(2*lfunc.NFBasis)
		RollData[:lfunc.NFBasis] = RealRoll*lfunc.ProfileFData[i][:lfunc.NFBasis]-ImagRoll*lfunc.ProfileFData[i][lfunc.NFBasis:]
		RollData[lfunc.NFBasis:] = ImagRoll*lfunc.ProfileFData[i][:lfunc.NFBasis]+RealRoll*lfunc.ProfileFData[i][lfunc.NFBasis:]
		

		MNM = np.dot(s[i], s[i])
		dNM = np.dot(RollData, s[i])
		MLAmp = dNM/MNM

		PSignal = MLAmp*s[i]

		Res=RollData-PSignal
		MLSigma = np.std(Res)


		x0[i] = MLAmp
		x0[i+lfunc.NToAs] = MLSigma

		RR = np.dot(Res, Res)

		AmpStep =  MNM/(MLSigma*MLSigma)
		SigmaStep = 3*RR/(MLSigma*MLSigma*MLSigma*MLSigma) - 2.0*lfunc.NFBasis/(MLSigma*MLSigma)


		cov_diag[i] = AmpStep
		cov_diag[i+lfunc.NToAs] = SigmaStep
		
		#Make Matrix for Linear Parameters
		LinearSize = 1  + lfunc.numTime + NCoeff*(lfunc.EvoNPoly+1) 

		HessMatrix = np.zeros([LinearSize, 2*lfunc.NFBasis])
		
		#Hessian for Phase parameter
		
		PhaseScale = -1*MLAmp/MLSigma
		LinCount = 0
		HessMatrix[LinCount,:] = PhaseScale*j[i]*lfunc.ReferencePeriod
		LinCount += 1
		
		#Hessian for Timing Model

		for c in range(lfunc.numTime):
			HessMatrix[LinCount, :] = j[i]*PhaseScale*lfunc.designMatrix[i,c]
			LinCount += 1
			

		#Hessian for Shapelet parameters
		
		fvals = ((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)

		ShapeBasis = InterpFBasis2[InterpBins[i]]

		for c in range(1, lfunc.TotCoeff):
			for p in range(lfunc.EvoNPoly+1):
				HessMatrix[LinCount, :] = fvals[p]*ShapeBasis[:,c]*MLAmp/MLSigma
				LinCount += 1
				
				
						
		OneHess = np.dot(HessMatrix, HessMatrix.T)
		
		#add phase prior to hessian
		OneHess[0,0]  += (1.0/lfunc.PhasePrior/lfunc.PhasePrior)/lfunc.NToAs
		
		cov_diag[lfunc.NToAs*2:] += OneHess.diagonal()
		hess_dense += OneHess
		
		

	if(diagonalGHS == False):		
		#Now do EVD on the dense part of the matrix
	
		V, M = sl.eigh(hess_dense)
	
		cov_diag[lfunc.NToAs*2:] = V
	
	else:
	
		hess_dense = np.eye(np.shape(hess_dense)[0])
		M = copy.copy(hess_dense)
		
	
	#Complete the start point by filling in extra parameters

	pcount = lfunc.NToAs*2
	
	x0[pcount] = lfunc.MeanPhase
	pcount += 1
	
	x0[pcount:pcount + lfunc.numTime] = 0
	pcount += lfunc.numTime
	
	if(lfunc.EvoNPoly == 0):
		x0[pcount:pcount + NCoeff*(lfunc.EvoNPoly+1)] = (lfunc.MLShapeCoeff[1:].T).flatten()[:lfunc.TotCoeff-1]
	else:
		x0[pcount:pcount + NCoeff*(lfunc.EvoNPoly+1)] = (lfunc.MLShapeCoeff[1:]).flatten()
		
	pcount += NCoeff*(lfunc.EvoNPoly+1)
		
			
	return x0, cov_diag, M, hess_dense
	
	


def FFTGHSLogLike(ndim, x, like, g):
	    
	    
	params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))
	
	
	#Send relevant parameters to physical coordinates for likelihood
	
	DenseParams = params[2*lfunc.NToAs:]
	PhysParams = np.dot(EigM, DenseParams)
	params[2*lfunc.NToAs:] = PhysParams
	#print("Phys Params: ", PhysParams)
	

	grad=np.zeros(ndim[0])

	pcount = 0

	ProfileAmps = params[pcount:pcount+lfunc.NToAs]
	pcount += lfunc.NToAs

	ProfileNoise = params[pcount:pcount+lfunc.NToAs]*params[pcount:pcount+lfunc.NToAs]
	pcount += lfunc.NToAs
	
	Phase = params[pcount]
	phasePrior = 0.5*(Phase-lfunc.MeanPhase)*(Phase-lfunc.MeanPhase)/lfunc.PhasePrior/lfunc.PhasePrior
	phasePriorGrad = 1*(Phase-lfunc.MeanPhase)/lfunc.PhasePrior/lfunc.PhasePrior
	pcount += 1
	
	TimingParameters = params[pcount:pcount+lfunc.numTime]
	pcount += lfunc.numTime
		
	NCoeff = lfunc.TotCoeff-1
	
	ShapeAmps=np.zeros([lfunc.TotCoeff, lfunc.EvoNPoly+1])
	ShapeAmps[0][0] = 1
	ShapeAmps[1:]=params[pcount:pcount + NCoeff*(lfunc.EvoNPoly+1)].reshape([NCoeff,(lfunc.EvoNPoly+1)])

	pcount += NCoeff*(lfunc.EvoNPoly+1)
	
	
	TimeSignal = np.dot(lfunc.designMatrix, TimingParameters)
	


	like[0] = 0

	xS = lfunc.ShiftedBinTimes[:,0] - Phase*lfunc.ReferencePeriod 
	
	if(lfunc.numTime>0):
			xS -= TimeSignal
			
	xS = ( xS + lfunc.ReferencePeriod/2) % (lfunc.ReferencePeriod ) - lfunc.ReferencePeriod/2

	InterpBins = (xS%(lfunc.ReferencePeriod/lfunc.Nbins[:])/lfunc.InterpolatedTime).astype(int)
	WBTs = xS-lfunc.InterpolatedTime*InterpBins
	RollBins=(np.round(WBTs/(lfunc.ReferencePeriod/lfunc.Nbins[:]))).astype(np.int)

	#ShapeAmps=lfunc.MLShapeCoeff

	s = np.sum([np.dot(InterpFBasis2[InterpBins], ShapeAmps[:,i])*(((lfunc.psr.freqs - lfunc.EvoRefFreq)/1000.0)**i).reshape(lfunc.NToAs,1) for i in range(lfunc.EvoNPoly+1)], axis=0)
	j = np.sum([np.dot(InterpJBasis2[InterpBins], ShapeAmps[:,i])*(((lfunc.psr.freqs - lfunc.EvoRefFreq)/1000.0)**i).reshape(lfunc.NToAs,1) for i in range(lfunc.EvoNPoly+1)], axis=0)


	for i in range(lfunc.NToAs):

		rfftfreqs=np.linspace(1,lfunc.NFBasis,lfunc.NFBasis)/lfunc.Nbins[i]
		RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
		ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

	
		RollData = np.zeros(2*lfunc.NFBasis)
		RollData[:lfunc.NFBasis] = RealRoll*lfunc.ProfileFData[i][:lfunc.NFBasis]-ImagRoll*lfunc.ProfileFData[i][lfunc.NFBasis:]
		RollData[lfunc.NFBasis:] = ImagRoll*lfunc.ProfileFData[i][:lfunc.NFBasis]+RealRoll*lfunc.ProfileFData[i][lfunc.NFBasis:]
		
		Res = RollData-s[i]*ProfileAmps[i]
		Chisq = np.dot(Res,Res)/ProfileNoise[i]
		
		AmpGrad = -1*np.dot(s[i], Res)/ProfileNoise[i]
		NoiseGrad = (-Chisq+2*lfunc.NFBasis)/np.sqrt(ProfileNoise[i])
		
		proflike = 0.5*Chisq + 0.5*2*lfunc.NFBasis*np.log(ProfileNoise[i])


		like[0] += proflike   
		
		grad[i] = AmpGrad
		grad[i+lfunc.NToAs] = NoiseGrad
		
		#Gradient for Phase
		pcount = 2*lfunc.NToAs
		
		PhaseGrad = np.dot(Res, j[i])*ProfileAmps[i]/ProfileNoise[i]
		grad[pcount] += PhaseGrad*lfunc.ReferencePeriod
		pcount += 1
		
		#Gradient for Timing Model
		TimeGrad = lfunc.designMatrix[i]*PhaseGrad
		grad[pcount:pcount+lfunc.numTime] += TimeGrad
		pcount += lfunc.numTime
		
		#Gradient for Shape Parameters
		ShapeGrad = np.dot(InterpFBasis2[InterpBins[i]].T, Res)/ProfileNoise[i]
		fvals = ((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)
		
		for c in range(1, lfunc.TotCoeff):
			for p in range(lfunc.EvoNPoly+1):
				grad[pcount] += -fvals[p]*ShapeGrad[c]*ProfileAmps[i]
				pcount += 1


	#Add phase prior to likelihood and gradient
	like[0] += phasePrior
	grad[2*lfunc.NToAs] += phasePriorGrad
	
	
	
	#Send relevant gradients to principle coordinates for sampling
	
	DenseGrad = copy.copy(grad[2*lfunc.NToAs:])
	PrincipleGrad = np.dot(EigM.T, DenseGrad)
	grad[2*lfunc.NToAs:] = PrincipleGrad
            
	#print("like:", like[0], "grad", PrincipleGrad, DenseGrad)
	for i in range(ndim[0]):
		g[i] = grad[i]



	return 



InterpFBasis2=np.zeros([4463, lfunc.NFBasis*2, lfunc.TotCoeff])
InterpFBasis2[:,lfunc.NFBasis:,:]=lfunc.InterpFBasis.imag
InterpFBasis2[:,:lfunc.NFBasis,:]=lfunc.InterpFBasis.real

InterpJBasis2=np.zeros([4463, lfunc.NFBasis*2, lfunc.TotCoeff])
InterpJBasis2[:,lfunc.NFBasis:,:]=lfunc.InterpFJitterMatrix.imag
InterpJBasis2[:,:lfunc.NFBasis,:]=lfunc.InterpFJitterMatrix.real



lfunc.PhasePrior = 1e-5
x0, cov_diag, EigM, hd = calculateGHSHessian(diagonalGHS=False)

DenseParams = x0[2*lfunc.NToAs:]
PrincipleParams = np.dot(EigM.T, DenseParams)
x0[2*lfunc.NToAs:] = PrincipleParams

GHS_resume = 0

if(GHS_resume == 0):
	test_gauss_outfile = open("test_outputShape3.dat", "w")
else:
	test_gauss_outfile = open("test_outputShape3.dat", "a")

file_prefix="testShape3"

run_guided_hmc( FFTGHSLogLike,
        			write_gauss_ghs_extract_with_logpostval,
        			lfunc.n_params, 
					x0.astype(np.float64),
        			(1.0/np.sqrt(cov_diag)).astype(np.float64),
        			file_prefix,
        			dim_scale_fact = 0.4,
        			max_steps = 10,
        			seed = -1,
        			resume = GHS_resume,
        			feedback_int = 100,
        			nburn = 100,
        			nsamp = 5000,
        			doMaxLike = 0)
        			
test_gauss_outfile.close()


chains=np.loadtxt("test_outputShape3.dat").T
	
