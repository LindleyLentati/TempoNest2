'''
from __future__ import absolute_import, unicode_literals, print_function
from ctypes import cdll
import sys

from ctypes import *
from numpy.ctypeslib import as_array
import signal
import inspect
import ctypes
'''
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
'''
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
import pycuda.driver as drv
import skcuda.fft as fft
import skcuda.linalg as cula
import skcuda.cublas as cublas

cula.init()
h = cublas.cublasCreate()
'''


'''
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
'''





lfunc = Likelihood()
lfunc.loadPulsar("OneChan.par", "T32F.tim", root='./results/GHS-1Chan-')


'''Get initial Fit to the Profile'''

lfunc.TScrunch(doplot = False, channels = 1)

lfunc.getInitialParams(MaxCoeff = 20, cov_diag=[0.01, 0.1, 0.1], resume=True, outDir = './InitFFTMNChains/All-Max20-', sampler='multinest', incScattering = False, mn_live = 1000,  fitNComps = 1, doplot = False)


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
	
	

'''
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
'''


InterpFBasis2=np.zeros([4463, lfunc.NFBasis*2, lfunc.TotCoeff])
InterpFBasis2[:,lfunc.NFBasis:,:]=lfunc.InterpFBasis.imag
InterpFBasis2[:,:lfunc.NFBasis,:]=lfunc.InterpFBasis.real

InterpJBasis2=np.zeros([4463, lfunc.NFBasis*2, lfunc.TotCoeff])
InterpJBasis2[:,lfunc.NFBasis:,:]=lfunc.InterpFJitterMatrix.imag
InterpJBasis2[:,:lfunc.NFBasis,:]=lfunc.InterpFJitterMatrix.real



lfunc.PhasePrior = 1e-5
x0, cov_diag, EigM, hd = calculateGHSHessian(diagonalGHS=False)
'''
DenseParams = x0[2*lfunc.NToAs:]
PrincipleParams = np.dot(EigM.T, DenseParams)
x0[2*lfunc.NToAs:] = PrincipleParams

GHS_resume = 0

if(GHS_resume == 0):
	test_gauss_outfile = open("./results/test_outputShape.dat", "w")
else:
	test_gauss_outfile = open("./results/test_outputShape.dat", "a")

file_prefix="./results/testShape"

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


chains=np.loadtxt("./results/test_outputShape.dat").T
'''

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import skcuda.cublas as cublas
h = cublas.cublasCreate()


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.
    """

    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)





mod = SourceModule("""

					#include <stdint.h>
					
                    __global__ void BinTimes(double *BinTimes, int32_t *NBins, double Phase, double RefPeriod, double InterpTime, double *xS, int32_t  *InterpBins, double *WBTs, int32_t *RollBins, uint64_t *InterpPointers, uint64_t *SomePointers, const int32_t NToAs)
                  {
                        const int i = blockDim.x*blockIdx.x + threadIdx.x;

						if(i < NToAs){
							xS[i] = BinTimes[i]  - Phase + RefPeriod*0.5; 

							xS[i] = xS[i] - trunc(xS[i]/RefPeriod)*RefPeriod; 
							xS[i] = xS[i]+RefPeriod - trunc((xS[i]+RefPeriod)/RefPeriod)*RefPeriod;
							xS[i] = xS[i] - RefPeriod*0.5;
						
							double InterpBin = xS[i] - trunc(xS[i]/(RefPeriod/NBins[i]))*(RefPeriod/NBins[i]);
							InterpBin = InterpBin + RefPeriod/NBins[i]  - trunc((InterpBin+RefPeriod/NBins[i])/(RefPeriod/NBins[i]))*(RefPeriod/NBins[i]);
							InterpBin /= InterpTime;
							InterpBins[i] = int(InterpBin);
						
							SomePointers[i] = InterpPointers[InterpBins[i]];
						
							WBTs[i] = xS[i]-InterpTime*InterpBins[i];
							RollBins[i] = int(round(WBTs[i]/(RefPeriod/NBins[i])));
						}
												
                   }

                    __global__ void PrepLikelihood(double *ProfAmps, double *ShapeAmps, const int32_t NToAs, const int32_t TotCoeff)
                  {
                        const int i = blockDim.x*blockIdx.x + threadIdx.x;
						//double freq = ToAFreqs[i];
						
						if(i < TotCoeff*NToAs){
							
							int index = i%TotCoeff;
							double amp = ShapeAmps[index];
					
							ProfAmps[i] = amp;
							
						}
						
                   }
                   
                   
                    __global__ void RotateData(double *data, double *freqs, const int32_t *RollBins, const int32_t *ToAIndex, double *RolledData, const int32_t Step)
                  {
                        const  int i = blockDim.x*blockIdx.x + threadIdx.x;

						if(i < Step){
							double freq = freqs[i];
							const int32_t  ToA_Index = ToAIndex[i];  
							const int32_t Roll = RollBins[ToA_Index];
							double RealRoll = cos(-2*M_PI*Roll*freq);
							double ImagRoll = sin(-2*M_PI*Roll*freq);
						
							RolledData[i] = RealRoll*data[i] - ImagRoll*data[i+Step];
							RolledData[i+Step] = ImagRoll*data[i] + RealRoll*data[i+Step];
						}
                   }
                   
                    __global__ void getRes(double *ResVec, double *NResVec, double *RolledData, double *Signal, double *Amps, double *Noise, const int32_t *ToAIndex, const int32_t *SignalIndex, const int32_t TotBins)
                  {
                        const int i = blockDim.x*blockIdx.x + threadIdx.x;

						if(i < TotBins){
		                    const int32_t ToA_Index = ToAIndex[i];  
							const int32_t Signal_Index = SignalIndex[i];  

						
							ResVec[i] = RolledData[i]-Amps[ToA_Index]*Signal[Signal_Index];
							NResVec[i] = ResVec[i]/Noise[ToA_Index];
						}
						
                   }
 					""")
 					
RotateData = mod.get_function("RotateData")
getRes = mod.get_function("getRes")
BinTimes = mod.get_function("BinTimes")
PrepLikelihood = mod.get_function("PrepLikelihood")

####################Transfer Matrices to GPU and allocate empty ones########

ProfAmps_GPU = gpuarray.empty((lfunc.NToAs, lfunc.TotCoeff, 1), np.float64)
ProfAmps_Pointer = bptrs(ProfAmps_GPU)

InterpFBasis_GPU = gpuarray.to_gpu(InterpFBasis2)
Interp_Pointers = np.array([InterpFBasis_GPU[i].ptr for i in range(len(InterpFBasis2))], dtype=np.uint64)
InterpPointers_GPU = gpuarray.to_gpu(Interp_Pointers)
i_arr_gpu = gpuarray.empty(lfunc.NToAs,  dtype=np.uint64)

gpu_ShiftedBinTimes = gpuarray.to_gpu((lfunc.ShiftedBinTimes[:,0]).astype(np.float64))
gpu_NBins =  gpuarray.to_gpu((lfunc.Nbins).astype(np.int32))

gpu_xS =  gpuarray.empty(lfunc.NToAs, np.float64) 
gpu_InterpBins =  gpuarray.empty(lfunc.NToAs, np.int32) 
gpu_WBTs =  gpuarray.empty(lfunc.NToAs, np.float64) 
gpu_RollBins =  gpuarray.empty(lfunc.NToAs, np.int32) 



Signal_GPU = gpuarray.empty((lfunc.NToAs, 2*lfunc.NFBasis, 1), np.float64)
Signal_Pointer = bptrs(Signal_GPU)

Flat_Data = np.zeros(2*lfunc.NToAs*lfunc.NFBasis)
Freqs = np.zeros(lfunc.NToAs*lfunc.NFBasis)
for i in range(lfunc.NToAs):
	Flat_Data[i*lfunc.NFBasis:(i+1)*lfunc.NFBasis] = lfunc.ProfileFData[i][:lfunc.NFBasis]
	Flat_Data[(lfunc.NToAs + i)*lfunc.NFBasis:(lfunc.NToAs + i+1)*lfunc.NFBasis] = lfunc.ProfileFData[i][lfunc.NFBasis:]
	
	Freqs[i*lfunc.NFBasis:(i+1)*lfunc.NFBasis] = np.linspace(1,lfunc.NFBasis,lfunc.NFBasis)/lfunc.Nbins[i]

gpu_Data = gpuarray.to_gpu(np.float64(Flat_Data))
gpu_RolledData =  gpuarray.empty(2*lfunc.NToAs*lfunc.NFBasis, np.float64)
gpu_Freqs = gpuarray.to_gpu(np.float64(Freqs)) 

ToA_Index = np.zeros(2*lfunc.NToAs*lfunc.NFBasis).astype(np.int32)
Signal_Index = np.zeros(2*lfunc.NToAs*lfunc.NFBasis).astype(np.int32)

for i in range(lfunc.NToAs):
	ToA_Index[i*lfunc.NFBasis:(i+1)*lfunc.NFBasis]=i
	ToA_Index[(lfunc.NToAs + i)*lfunc.NFBasis:(lfunc.NToAs + i+1)*lfunc.NFBasis]=i
	
	Signal_Index[i*lfunc.NFBasis:(i+1)*lfunc.NFBasis] = np.arange(2*i*lfunc.NFBasis,(2*i+1)*lfunc.NFBasis)
	Signal_Index[(lfunc.NToAs + i)*lfunc.NFBasis:(lfunc.NToAs + i+1)*lfunc.NFBasis] = np.arange((2*i+1)*lfunc.NFBasis,(2*i+2)*lfunc.NFBasis)
	
	
	
gpu_ToAIndex = gpuarray.to_gpu((ToA_Index).astype(np.int32))
gpu_SignalIndex = gpuarray.to_gpu((Signal_Index).astype(np.int32))

gpu_ResVec =  gpuarray.empty(2*lfunc.NToAs*lfunc.NFBasis, np.float64)
gpu_NResVec =  gpuarray.empty(2*lfunc.NToAs*lfunc.NFBasis, np.float64) 











def SimpleGPULike(x):


	####################Get Parameters########################################
			
			
	params = x0

	pcount = 0

	ProfileAmps = params[pcount:pcount+lfunc.NToAs]
	pcount += lfunc.NToAs

	ProfileNoise = params[pcount:pcount+lfunc.NToAs]*x0[pcount:pcount+lfunc.NToAs]
	pcount += lfunc.NToAs

	gpu_Amps = gpuarray.to_gpu(np.float64(ProfileAmps))
	gpu_Noise = gpuarray.to_gpu(np.float64(ProfileNoise)) 

	Phase = params[pcount]
	pcount += 1

	pcount += lfunc.numTime

	NCoeff = lfunc.TotCoeff-1

	ShapeAmps=np.zeros([lfunc.TotCoeff, lfunc.EvoNPoly+1])
	ShapeAmps[0][0] = 1
	ShapeAmps[1:]=params[pcount:pcount + NCoeff*(lfunc.EvoNPoly+1)].reshape([NCoeff,(lfunc.EvoNPoly+1)])

	ShapeAmps_GPU = gpuarray.to_gpu(ShapeAmps)

	pcount += NCoeff*(lfunc.EvoNPoly+1)


	####################Calculate Profile Amplitudes########################################

	block_size = 128
	grid_size = int(np.ceil(lfunc.TotCoeff*lfunc.NToAs*1.0/block_size))
	PrepLikelihood(ProfAmps_GPU, ShapeAmps_GPU, np.int32(lfunc.NToAs), np.int32(lfunc.TotCoeff), grid=(grid_size,1), block=(block_size,1,1))


	####################Calculate Phase Offsets########################################


	block_size = 128
	grid_size = int(np.ceil(lfunc.NToAs*1.0/block_size))
	BinTimes(gpu_ShiftedBinTimes, gpu_NBins, np.float64(Phase*lfunc.ReferencePeriod), np.float64(lfunc.ReferencePeriod), np.float64(lfunc.InterpolatedTime), gpu_xS, gpu_InterpBins, gpu_WBTs, gpu_RollBins, InterpPointers_GPU, i_arr_gpu, np.int32(lfunc.NToAs),  grid=(grid_size,1), block=(block_size,1,1))
		
	

	#InterpBins=gpu_InterpBins.get()

	#i_arr = np.array([InterpFBasis_GPU[i].ptr for i in InterpBins], dtype=np.uint64)
	#i_arr_gpu = gpuarray.to_gpu(i_arr)
	

	####################GPU Batch submit DGEMM for profile and jitter########################################

	alpha = np.float64(1.0)
	beta = np.float64(0.0)


	cublas.cublasDgemmBatched(h, 'n','n', 1, 2*lfunc.NFBasis, lfunc.TotCoeff, alpha, ProfAmps_Pointer.gpudata, 1, i_arr_gpu.gpudata, lfunc.TotCoeff, beta, Signal_Pointer.gpudata, 1, lfunc.NToAs)

	####################Rotate Data########################################
	
	block_size = 128
	grid_size = int(np.ceil(lfunc.NToAs*lfunc.NFBasis*1.0/block_size))
	Step = np.int32(lfunc.NToAs*lfunc.NFBasis)

	RotateData(gpu_Data,  gpu_Freqs, gpu_RollBins, gpu_ToAIndex, gpu_RolledData, Step, grid=(grid_size,1), block=(block_size,1,1))



	####################Compute Chisq########################################

	TotBins = np.int32(2*lfunc.NToAs*lfunc.NFBasis)
	grid_size = int(np.ceil(lfunc.NToAs*lfunc.NFBasis*2.0/block_size))	

	getRes(gpu_ResVec, gpu_NResVec, gpu_RolledData, Signal_GPU, gpu_Amps, gpu_Noise, gpu_ToAIndex, gpu_SignalIndex, TotBins, grid=(grid_size,1), block=(block_size,1,1))

	gpu_Chisq = cublas.cublasDdot(h, gpu_ResVec.size, gpu_ResVec.gpudata, 1, gpu_NResVec.gpudata, 1)

	gpulike = 0.5*gpu_Chisq + 0.5*2*lfunc.NFBasis*np.sum(np.log(ProfileNoise))


	return gpulike
	
	
def SimpleCPULike(params):
	    

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
	
	like = 0

	xS = lfunc.ShiftedBinTimes[:,0] - Phase*lfunc.ReferencePeriod 
			
	xS = ( xS + lfunc.ReferencePeriod/2) % (lfunc.ReferencePeriod ) - lfunc.ReferencePeriod/2

	InterpBins = (xS%(lfunc.ReferencePeriod/lfunc.Nbins[:])/lfunc.InterpolatedTime).astype(int)
	WBTs = xS-lfunc.InterpolatedTime*InterpBins
	RollBins=(np.round(WBTs/(lfunc.ReferencePeriod/lfunc.Nbins[:]))).astype(np.int)

	s = np.sum([np.dot(InterpFBasis2[InterpBins], ShapeAmps[:,i])*(((lfunc.psr.freqs - lfunc.EvoRefFreq)/1000.0)**i).reshape(lfunc.NToAs,1) for i in range(lfunc.EvoNPoly+1)], axis=0)

	for i in range(lfunc.NToAs):

		rfftfreqs=np.linspace(1,lfunc.NFBasis,lfunc.NFBasis)/lfunc.Nbins[i]
		RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
		ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

	
		RollData = np.zeros(2*lfunc.NFBasis)
		RollData[:lfunc.NFBasis] = RealRoll*lfunc.ProfileFData[i][:lfunc.NFBasis]-ImagRoll*lfunc.ProfileFData[i][lfunc.NFBasis:]
		RollData[lfunc.NFBasis:] = ImagRoll*lfunc.ProfileFData[i][:lfunc.NFBasis]+RealRoll*lfunc.ProfileFData[i][lfunc.NFBasis:]
		
		Res = RollData-s[i]*ProfileAmps[i]
		Chisq = np.dot(Res,Res)/ProfileNoise[i]

		
		proflike = 0.5*Chisq + 0.5*2*lfunc.NFBasis*np.log(ProfileNoise[i])


		like += proflike   






	return like


step_size=1.0/np.sqrt(cov_diag)

import time

print "running gpu test: 10k calls"
ltot = 0
start = time.time()
for i in range(10000):
	x = x0 + np.random.normal(0,1,lfunc.n_params)*step_size	
	l = SimpleGPULike(x)
	ltot += l
stop=time.time()
print "run time: ", stop-start
print "running cpu test: 10k calls"
ltot = 0
start = time.time()
for i in range(10000):
	x = x0 + np.random.normal(0,1,lfunc.n_params)*step_size	
	l = SimpleCPULike(x)
	ltot += l
stop=time.time()
print "run time: ", stop-start	

