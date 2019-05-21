from libstempo.libstempo import *
import libstempo as T
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
from scipy import optimize
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
import scipy as sp
import corner
import pymultinest
import math
import os
import threading, subprocess
import pickle
import copy as copy
import time
import ghs

HaveGPUS = False
try:
	import pycuda.autoinit
	import pycuda.gpuarray as gpuarray
	from pycuda.compiler import SourceModule
	import skcuda.cublas as cublas
	HaveGPUS = True
except:
	print "GPU modules not available (or broken)  :( \n"


class Likelihood(object):
    
	def __init__(self, useGPU = False):
	
		

		self.useGPU = useGPU
		self.InitGPU = True

		self.SECDAY = 24*60*60

		self.parfile = None
		self.timfile = None
		self.root = None
		self.psr = None  
		self.SatSecs = None
		self.SatDays = None
		self.SSBFreqs = None
		self.FNames = None
		self.NToAs = None
		self.numTime = None	   
		self.TempoPriors = None
		self.ArchiveMap = None

		self.ProfileData = None
		self.ProfileFData = None
		self.fftFreqs = None
		self.NFBasis = None
		self.ProfileMJDs = None
		self.ProfileInfo = None
		self.ChansPerEpoch = None
		self.NumEpochs = None

		self.toas= None
		self.OrigSats = None
		self.residuals =  None
		self.BatCorrs =  None
		self.ModelBats =  None

		self.designMatrix = None
		self.FisherU = None
		self.FisherS = None

		self.TScrunched = None
		self.TScrunchedNoise = None
		self.TScrunchedFreqs = None

		self.Nbins = None
		self.ShiftedBinTimes = None
		self.ReferencePeriod = None
		self.ProfileStartBats = None
		self.ProfileEndBats = None
		self.FoldingPeriods = None

		self.MaxCoeff = None	
		self.TotCoeff = None
		self.MLShapeCoeff = None
		self.MeanBeta = None
		self.MeanPhase = None
		self.MeanScatter = None
		self.PhasePrior = None
		self.CompSeps = None

		self.doplot = None
		self.ReturnProfile = False

		self.n_params = None
		self.parameters = None
		self.pmin = None
		self.pmax = None
		self.startPoint = None
		self.cov_diag = None
		self.hess = None
		self.EigM = None
		self.EigV = None
		self.BLNHess = None
		self.BLNEigM = None
		self.GHSoutfile = None

		self.InterpolatedTime = None
		self.InterpBasis = None
		self.InterpJitterMatrix = None

		self.InterpFBasis = None
		self.InterpJBasis = None
		self.OneFBasis = None


		self.getShapeletStepSize = False
		self.TScrunchChans = None

		self.EvoRefFreq = 1400.0
		self.EvoNPoly = 0
		self.TScrunchShapeErr = None

		self.chains = None
		self.ShapePhaseCov = None
		self.returnVal  = 0;

		self.FindML = False

		#Model Parameters


		self.DenseParams = 0	
		self.DiagParams = 0
		self.LinearParams = 0
		self.MLParameters = []
		self.parameters = []
		self.ParametersToWrite = []
		self.ParamDict = {}


		self.fitNCoeff = False
		self.fitNComps = False
		self.NScatterEpochs = 0
		self.ScatterInfo = None
		self.ScatterRefFreq = None

		self.incPNoise = False
		self.fitPNoise = False
		self.incPAmps = False
		self.fitPAmps = False
		self.incPhase = False
		self.fitPhase = False
		self.incLinearTM = False
		self.fitLinearTM = False
		self.incProfile = False
		self.fitProfile = False
		self.incScatter = False
		self.fitScatter = False
		self.fitScatterFreqScale = False
		self.fitScatterPrior = 0
		self.incEQUAD = False
		self.fitEQUADSignal = False
		self.fitEQUADPrior = False
		self.incECORR = False
		self.fitECORRSignal = False
		self.fitECORRPrior = False
		self.incBaselineNoise = False
		self.BaselineNoisePrior = None
		self.fitBaselineNoiseAmpPrior = False
		self.fitBaselineNoiseSpecPrior = False
		self.BaselineNoiseParams = None
		self.BaselineNoiseRefFreq = 1



		if(self.useGPU == True):

			self.CUhandle = cublas.cublasCreate()

			mod = SourceModule("""

			#include <stdint.h>

			__global__ void BinTimes(double *BinTimes, int32_t *NBins, double Phase, double *TimeSignal, double *JitterSignal, double *RefPeriods, double InterpTime, double *xS, int32_t  *InterpBins, double *WBTs, int32_t *RollBins, uint64_t *InterpPointers, uint64_t *SomePointers, uint64_t *InterpJPointers, uint64_t *SomeJPointers, const int32_t NToAs, const int32_t InterpSize){

				const int i = blockDim.x*blockIdx.x + threadIdx.x;

				if(i < NToAs){
					double RefPeriod = RefPeriods[i];
					double OneBin = RefPeriod/NBins[i];

					xS[i] = BinTimes[i]  - Phase - TimeSignal[i] - JitterSignal[i] + RefPeriod*0.5; 

					xS[i] = xS[i] - trunc(xS[i]/RefPeriod)*RefPeriod; 
					xS[i] = xS[i] + RefPeriod - trunc((xS[i]+RefPeriod)/RefPeriod)*RefPeriod;
					xS[i] = xS[i] - RefPeriod*0.5;

					double InterpBin = -xS[i] - trunc(-xS[i]/OneBin)*OneBin;
					InterpBin = InterpBin + OneBin - trunc((InterpBin+OneBin)/OneBin)*OneBin;
					InterpBin /= InterpTime;
					InterpBins[i] = int(floor(InterpBin+0.5))%InterpSize;

					SomePointers[i] = InterpPointers[InterpBins[i]];

					SomeJPointers[i] = InterpJPointers[InterpBins[i]];

					WBTs[i] = xS[i] + InterpTime*(InterpBins[i]-1);
					RollBins[i] = int(floor(WBTs[i]/OneBin + 0.5));
				}

			}

			__global__ void PrepLikelihood(double *ProfAmps, double *ShapeAmps, double *ToAFreqs, const int32_t NToAs, const int32_t TotCoeff, const int32_t NEvoPoly, double RefFreq){

				const int i = blockDim.x*blockIdx.x + threadIdx.x;
				
				if(i < TotCoeff*NToAs){
					int j = 0;
					int index = ((NEvoPoly+1)*i)%(TotCoeff*(NEvoPoly+1));
					int ToA_Index = trunc(i*1.0/TotCoeff);
					double FreqFactor = (ToAFreqs[ToA_Index]/1000000.0 - RefFreq)/1000.0;
					double PolyFactor = 1;
					
					double EvoAmp = 0;
					for(j = 0; j < NEvoPoly+1; j++){
						
						EvoAmp += ShapeAmps[index+j]*PolyFactor;
						PolyFactor *= FreqFactor;
					}

					ProfAmps[i] = EvoAmp;

				}

			}


			__global__ void RotateData(double *data, double *freqs, const int32_t *RollBins, const int32_t *ToAIndex, double *RolledData, const int32_t Step){

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

			__global__ void getRes(double *ResVec, double *NResVec, double *RolledData, double *Signal, double *Amps, double *Noise, const int32_t *ToAIndex, const int32_t *SignalIndex, const int32_t TotBins){

				const int i = blockDim.x*blockIdx.x + threadIdx.x;

				if(i < TotBins){
					const int32_t ToA_Index = ToAIndex[i];  
					const int32_t Signal_Index = SignalIndex[i];  

					ResVec[Signal_Index] = RolledData[i]-Amps[ToA_Index]*Signal[Signal_Index];
					NResVec[Signal_Index] = ResVec[Signal_Index]/Noise[ToA_Index];
				}

			}

			__global__ void getBaselineNoiseRes(double *ResVec, double *NResVec, double *RolledData, double *Signal, double *Amps, double *Noise, const int32_t *ToAIndex, const int32_t *SignalIndex, const int32_t TotBins, const int32_t NFBasis, double *BLAmps, double *BLSpecs, double *PriorDet, const int32_t BLRefP){

				const int i = blockDim.x*blockIdx.x + threadIdx.x;

				if(i < TotBins){
					const int32_t ToA_Index = ToAIndex[i];  
					const int32_t Signal_Index = SignalIndex[i]; 
					const int32_t F_Index = Signal_Index%NFBasis;

					double BLAmp = pow(10.0, 2*BLAmps[ToA_Index]);
					double BLSpec = BLSpecs[ToA_Index]; 
					double Freq = (F_Index + 1.0)/BLRefP;

					if(NFBasis-F_Index <= 5){
						BLAmp = 0;
					}

					double NoiseVal = BLAmp*pow(Freq, -BLSpec) + Noise[ToA_Index];
					//double NDet = log(NoiseVal);

					ResVec[Signal_Index] = RolledData[i]-Amps[ToA_Index]*Signal[Signal_Index];
					NResVec[Signal_Index] = ResVec[Signal_Index]/NoiseVal;
					//PriorDet[Signal_Index] = NDet;

				}

			}


			__global__ void getBaselineNoiseRes2(double *ResVec, double *NResVec, double *RolledData, double *Signal, double *Amps, double *Noise, const int32_t *ToAIndex, const int32_t *SignalIndex, const int32_t TotBins, const int32_t NFBasis, double *BLAmps, double *BLSpecs, double *PriorDet, const int32_t BLRefP, const int32_t NToAs){

				const int i = blockDim.x*blockIdx.x + threadIdx.x;

				if(i < NToAs){

					int32_t c = 0;

                                        PriorDet[i] = 0;

                                        double Amp = pow(10.0, 2*BLAmps[i]);
                                        double Spec = BLSpecs[i];
					double NVal = Noise[i];

					double NoiseGrad = 0;
                                        double AmpGrad = 0;
                                        double SpecGrad = 0;

                                        for(c = 0; c < NFBasis; c++){

						double BLAmp = pow(10.0, 2*BLAmps[i]);
						double BLSpec = BLSpecs[i]; 
						double Freq = (c + 1.0)/BLRefP;

						if(NFBasis-c <= 5){
							BLAmp = 0;
						}

						double BLNPower = BLAmp*pow(Freq, -BLSpec);
						double NoiseVal = BLNPower + NVal;
						double Denom = (1.0 - ResVec[i*2*NFBasis + c]*NResVec[i*2*NFBasis + c])/NoiseVal;

						NoiseGrad += sqrt(Noise[i])*Denom;
						AmpGrad +=   log(10.0)*BLNPower*Denom;
						SpecGrad += -0.5*log(Freq)*BLNPower*Denom;


						Denom = (1.0 - ResVec[i*2*NFBasis + c + NFBasis]*NResVec[i*2*NFBasis + c + NFBasis])/NoiseVal;

						NoiseGrad += sqrt(Noise[i])*Denom;
						AmpGrad +=   log(10.0)*BLNPower*Denom;
						SpecGrad += -0.5*log(Freq)*BLNPower*Denom;

						double NDet = log(NoiseVal);

						PriorDet[i] += 2*NDet;
						
					}
					BLAmps[i] = AmpGrad;
					BLSpecs[i] = SpecGrad;
					Noise[i] = NoiseGrad;

				}

			}

			__global__ void getBaselineNoiseGrads(double *NResVec, double *BaselineNoise, double *Amps, double *Specs, double *PriorLike, const int32_t Step, const int32_t NToAs, const int32_t NFBasis, const int32_t BLRefP){

				const int i = blockDim.x*blockIdx.x + threadIdx.x;

				if(i < NToAs){

					int32_t c = 0;

					PriorLike[i] = 0;

					double Amp = pow(10.0, 2*Amps[i]);
					double Spec = Specs[i];

					Amps[i] = 0;
					Specs[i] = 0;

					for(c = 0; c < NFBasis; c++){

						double freq = ((c+1.0)/BLRefP);
						double power = Amp*pow(freq, -Spec);
						double pdet = 0.5*log(power);

						PriorLike[i] += 0.5*BaselineNoise[i*2*NFBasis+c]*BaselineNoise[i*2*NFBasis+c]/power + pdet;
						PriorLike[i] += 0.5*BaselineNoise[i*2*NFBasis+c+NFBasis]*BaselineNoise[i*2*NFBasis+c+NFBasis]/power + pdet;

						Amps[i] += log(10.0)*(-BaselineNoise[i*2*NFBasis+c]*BaselineNoise[i*2*NFBasis+c]/power + 1);
						Amps[i] += log(10.0)*(-BaselineNoise[i*2*NFBasis+c+NFBasis]*BaselineNoise[i*2*NFBasis+c+NFBasis]/power + 1);

						Specs[i] += 0.5*log(freq)*(BaselineNoise[i*2*NFBasis+c]*BaselineNoise[i*2*NFBasis+c]/power - 1);
				                Specs[i] += 0.5*log(freq)*(BaselineNoise[i*2*NFBasis+c+NFBasis]*BaselineNoise[i*2*NFBasis+c+NFBasis]/power - 1);						

						BaselineNoise[i*2*NFBasis+c] = -NResVec[i*2*NFBasis+c] + BaselineNoise[i*2*NFBasis+c]/power;

						BaselineNoise[i*2*NFBasis+c+NFBasis] = -NResVec[i*2*NFBasis+c+NFBasis] + BaselineNoise[i*2*NFBasis+c+NFBasis]/power;
						
						
					}
				}

			}


			__global__ void Scatter(double *SignalVec, double *JitterVec, double *ScatterGrad, double *ScatterParameters, double *freqs, double *ToA_Freqs, const int32_t Step, const int32_t *ScatterIndex, const int32_t *ToAIndex, const int32_t *SignalIndex, double *ProfileAmps, const int32_t *NBins, double *ReferencePeriods, const int32_t NFBasis, const int32_t TotCoeff, double *ScatterBasis, double *InterpBasis, const int32_t *InterpBins, double FreqScale, double ScatterRefFreq){

				const  int i = blockDim.x*blockIdx.x + threadIdx.x;
	
				if(i < Step){

					const int32_t  ToA_Index = ToAIndex[i];  
					const int32_t  RSignal_Index = SignalIndex[i];
					const int32_t  ISignal_Index = SignalIndex[i+Step];

					double ReferencePeriod = ReferencePeriods[ToA_Index];

					double ToA_freq = ToA_Freqs[ToA_Index]/ScatterRefFreq;
					//double ToA_freq = ToA_Freqs[ToA_Index]/pow(10.0, 9);
					ToA_freq = pow(ToA_freq, FreqScale);


					double tau = ScatterParameters[ScatterIndex[i]];
					double freq = freqs[i]*NBins[ToA_Index]/ReferencePeriod;
					double w = 2*M_PI*freq;
					tau = tau/ToA_freq;

					double RConv = 1.0/(w*w*tau*tau+1); 
					double IConv = -1*w*tau/(w*w*tau*tau+1);

					double RProf = SignalVec[RSignal_Index];
					double IProf = SignalVec[ISignal_Index];

					double RConfProf = RConv*RProf - IConv*IProf;
					double IConfProf = IConv*RProf + RConv*IProf;

					double PAmp = ProfileAmps[ToA_Index];
					double GradDenom = 1.0/((1.0 + tau*tau*w*w)*(1.0 + tau*tau*w*w));
					double RealSGrad = 2*tau*tau*w*w*log(10.0)*GradDenom*RProf*PAmp + tau*w*(tau*tau*w*w - 1)*log(10.0)*GradDenom*IProf*PAmp;
					double ImagSGrad = 2*tau*tau*w*w*log(10.0)*GradDenom*IProf*PAmp - tau*w*(tau*tau*w*w - 1)*log(10.0)*GradDenom*RProf*PAmp;
					ScatterGrad[RSignal_Index] = RealSGrad;
					ScatterGrad[ISignal_Index] = ImagSGrad;	

					SignalVec[RSignal_Index] = RConfProf;
					SignalVec[ISignal_Index] = IConfProf;

					RConfProf = RConv*JitterVec[RSignal_Index] - IConv*JitterVec[ISignal_Index];
					IConfProf = IConv*JitterVec[RSignal_Index] + RConv*JitterVec[ISignal_Index];

					JitterVec[RSignal_Index] = RConfProf;
					JitterVec[ISignal_Index] = IConfProf;
		
					const int32_t InterpIndex = InterpBins[ToA_Index];
					int32_t c = 0;
					for(c = 0; c < TotCoeff; c++){
		
						RProf = InterpBasis[InterpIndex*2*NFBasis*TotCoeff + (i-ToA_Index*NFBasis)*TotCoeff + c];
						IProf = InterpBasis[InterpIndex*2*NFBasis*TotCoeff + NFBasis*TotCoeff + (i-ToA_Index*NFBasis)*TotCoeff + c];
			
						RConfProf = RConv*RProf - IConv*IProf;
						IConfProf = IConv*RProf + RConv*IProf;			
			
						ScatterBasis[RSignal_Index*TotCoeff+c] = RConfProf;
						ScatterBasis[ISignal_Index*TotCoeff+c] = IConfProf;
					}

				}
			}
			""")

			self.GPURotateData = mod.get_function("RotateData")
			self.GPUGetRes = mod.get_function("getRes")
			self.GPUBinTimes = mod.get_function("BinTimes")
			self.GPUPrepLikelihood = mod.get_function("PrepLikelihood")
			self.GPUScatter = mod.get_function("Scatter")
			self.GPUGetBaselineRes = mod.get_function("getBaselineNoiseRes")
			self.GPUGetBaselineGrads = mod.get_function("getBaselineNoiseRes2")


	import time, sys

	# update_progress() : Displays or updates a console progress bar
	## Accepts a float between 0 and 1. Any int will be converted to a float.
	## A value under 0 represents a 'halt'.
	## A value at 1 or bigger represents 100%
	def update_progress(self, progress):
	    barLength = 10 # Modify this to change the length of the progress bar
	    status = ""
	    if isinstance(progress, int):
		progress = float(progress)
	    if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	    if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	    if progress >= 1:
		progress = 1
		status = "Done...\r\n"
	    block = int(round(barLength*progress))
	    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
	    sys.stdout.write(text)
	    sys.stdout.flush()

	
	def loadPulsar(self, parfile, timfile, ToPickle = False, FromPickle = False, root='Example', iters=1, usePreFit = False):

		self.root = root
		self.psr = T.tempopulsar(parfile=parfile, timfile = timfile)    
	
		if(usePreFit == True):
			self.psr.toas()
			self.psr.residuals()		
		else:
			if(iters > 0):
				self.psr.fit(iters=iters)

		self.SatSecs = self.psr.satSec()
		self.SatDays = self.psr.satDay()
		self.FNames = self.psr.filename()
		self.NToAs = self.psr.nobs
		self.SSBFreqs = self.psr.ssbfreqs()	    


		#Check how many timing model parameters we are fitting for (in addition to phase)
		self.numTime=len(self.psr.pars())
		redChisq = self.psr.chisq()/(self.psr.nobs-len(self.psr.pars())-1)
		self.TempoPriors=np.zeros([self.numTime,2]).astype(np.float128)
		for i in range(self.numTime):
			self.TempoPriors[i][0]=self.psr[self.psr.pars()[i]].val
			self.TempoPriors[i][1]=self.psr[self.psr.pars()[i]].err/np.sqrt(redChisq)
			print "fitting for: ", self.psr.pars()[i], self.TempoPriors[i][0], self.TempoPriors[i][1]

		#Now loop through archives, and work out what subint/frequency channel is associated with a ToA.
		#Store whatever meta data is needed (MJD of the bins etc)
		#If multiple polarisations are present we first PScrunch.

		self.ProfileData=[]
		self.ProfileMJDs=[]
		self.ProfileInfo=[]
		self.ChansPerEpoch=[]

		if(FromPickle == False):

			self.ArchiveMap = np.zeros([self.NToAs, 2])
			profcount = 0
			while(profcount < self.NToAs):

			    self.update_progress(np.float64(profcount)/self.NToAs)
			    arch=psrchive.Archive_load(self.FNames[profcount])

			    
			    npol = arch.get_npol()
			    if(npol>1):
				arch.pscrunch()
			    arch.remove_baseline()
			    nsub=arch.get_nsubint()


			    for i in range(nsub):

				if(profcount == self.NToAs):
					break

				subint=arch.get_Integration(i)
			
				nbins = subint.get_nbin()
				nchans = subint.get_nchan()
				npols = subint.get_npol()
				foldingperiod = subint.get_folding_period()
				inttime = subint.get_duration()
				centerfreq = subint.get_centre_frequency()
			
				#print "Info:", profcount, i, nbins, nchans, npols, foldingperiod, inttime, centerfreq
			
				firstbin = subint.get_epoch()
				intday = firstbin.intday()
				fracday = firstbin.fracday()
				intsec = firstbin.get_secs()
				fracsecs = firstbin.get_fracsec()
				isdedispersed = subint.get_dedispersed()
			
				pulsesamplerate = foldingperiod/nbins/self.SECDAY;
			
				nfreq=subint.get_nchan()
			
				FirstBinSec = intsec + np.float128(fracsecs)
				SubIntTimeDiff = FirstBinSec-self.SatSecs[profcount]*self.SECDAY
				PeriodDiff = SubIntTimeDiff*self.psr['F0'].val
			
				usedChan = []
				if(abs(PeriodDiff) < 2.0):
					for j in range(nfreq):
						chanfreq = subint.get_centre_frequency(j)
						toafreq = self.psr.freqs[profcount]
						prof=subint.get_Profile(0,j)
						profamps = prof.get_amps()

						if(np.sum(profamps) != 0 and abs(toafreq-chanfreq) < 0.001):
							noiselevel=self.GetProfNoise(profamps)
							self.ProfileData.append(np.copy(profamps))

							self.ProfileInfo.append([self.SatSecs[profcount], self.SatDays[profcount], np.float128(intsec)+np.float128(fracsecs), pulsesamplerate, nbins, foldingperiod, noiselevel])                    
							#print "ChanInfo:", j, chanfreq, toafreq, np.sum(profamps), profcount

							usedChan.append(profcount)
							self.ArchiveMap[profcount, 0] = i
							self.ArchiveMap[profcount, 1] = j
							profcount += 1
							
							if(profcount == self.NToAs):
								break

					self.ChansPerEpoch.append(usedChan)


			self.ProfileInfo=np.array(self.ProfileInfo)
			self.ProfileData=np.array(self.ProfileData)
			self.ChansPerEpoch = np.array(self.ChansPerEpoch)
			self.NumEpochs = len(self.ChansPerEpoch)

			if(ToPickle == True):
				print "Pickling Data"
				output = open(self.root+'-ProfData.pickle', 'wb')
				pickle.dump(self.ProfileData, output)
				pickle.dump(self.ProfileInfo, output)
				output.close()

		if(FromPickle == True):
			print "Loading from Pickled Data"
			pick = open(self.root+'-ProfData.pickle', 'rb')
			self.ProfileData = pickle.load(pick)
			self.ProfileInfo = pickle.load(pick)
			pick.close()

		self.toas=self.psr.toas()
		self.residuals = self.psr.residuals(removemean=False)
		self.BatCorrs = self.psr.batCorrs()
		self.OrigSats = copy.copy(self.psr.stoas)
		self.ModelBats = self.psr.satSec() + self.BatCorrs - self.residuals/self.SECDAY
		#self.FoldingPeriods = np.ones(self.NToAs)*self.ReferencePeriod #self.ProfileInfo[:,5]


		#get design matrix for linear timing model, setup jump proposal

		if(self.numTime > 0):
			self.designMatrix=self.psr.designmatrix(incoffset=False)
			for i in range(self.numTime):
				self.designMatrix[:,i] *= self.TempoPriors[i][1]
				zval = self.designMatrix[0,i]
				self.designMatrix[:,i] -= zval

			self.designMatrix=np.float64(self.designMatrix)
			N=np.diag(1.0/(self.psr.toaerrs*10.0**-6))
			Fisher=np.dot(self.designMatrix.T, np.dot(N, self.designMatrix))
			FisherU,FisherS,FisherVT=np.linalg.svd(Fisher)

			self.FisherU = FisherU
			self.FisherS = FisherS

		'''
		print "making fake pulsars"
		temppsr = T.tempopulsar(parfile=parfile, timfile = timfile)
		newSec = self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0.5
		newSat = np.floor(temppsr.stoas)+newSec
		temppsr.stoas[:] = newSat
		temppsr.formbats()
		tempBatCorrs = temppsr.batCorrs()

		temppsr2 = T.tempopulsar(parfile=parfile, timfile = timfile)
		newSec2 = self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*(self.ProfileInfo[:,4]-1) + self.ProfileInfo[:,3]*0.5
		newSat2 = np.floor(temppsr2.stoas)+newSec2
		temppsr2.stoas[:] = newSat2
		temppsr2.formbats()
		tempBatCorrs2 = temppsr2.batCorrs()
		'''
		'''
		self.ProfileStartBats = self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0 + self.ProfileInfo[:,3]*0.5 + tempBatCorrs # self.BatCorrs
		self.ProfileEndBats =  self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*(self.ProfileInfo[:,4]-1) + self.ProfileInfo[:,3]*0.5 + tempBatCorrs2 #self.BatCorrs
		'''
	
		self.ProfileStartBats = self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0 + self.ProfileInfo[:,3]*0.5 + self.BatCorrs
		self.ProfileEndBats =  self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*(self.ProfileInfo[:,4]-1) + self.ProfileInfo[:,3]*0.5 + self.BatCorrs

		#self.FoldingPeriods = (self.ProfileEndBats - self.ProfileStartBats)*self.SECDAY  #self.ProfileInfo[:,5]
		self.FoldingPeriods = self.ProfileInfo[:,5]

		self.Nbins = (self.ProfileInfo[:,4]).astype(int)
		ProfileBinTimes = []
		for i in range(self.NToAs):
			ProfileBinTimes.append(((np.linspace(self.ProfileStartBats[i], self.ProfileEndBats[i], self.Nbins[i]) - self.ModelBats[i])*self.SECDAY)[0])
		self.ShiftedBinTimes = np.float64(np.array(ProfileBinTimes))
		#self.ReferencePeriod = np.float64(np.mean(self.FoldingPeriods))
		self.ReferencePeriod = np.float64(self.ProfileInfo[0][5])



	
	#Funtion to determine an estimate of the white noise in the profile data
	def GetProfNoise(self, profamps):

		Nbins = len(profamps)
		Step=100
		noiselist=[]
		for i in range(Nbins-Step):
			noise=np.std(profamps[i:i+Step])
			noiselist.append(noise)
		noiselist=np.array(noiselist)
		minnoise=np.min(noiselist)
		threesiglist=noiselist[noiselist<3*minnoise]
		mediannoise=np.median(threesiglist)
		return mediannoise



	def TScrunch(self, doplot=True, channels=None, ChanSep = None, FreqRange = None, FromPickle = False, ToPickle = False, FromTM = False):


		minfreq = np.min(self.psr.freqs)
		maxfreq = np.max(self.psr.freqs)

		weights = 1.0/self.psr.toaerrs**2


		zipped=np.zeros([self.NToAs,2])
		zipped[:,0]=self.psr.freqs
		zipped[:,1]=weights
		zipped=zipped[zipped[:,0].argsort()]

		uniquezipped = np.zeros([len(np.unique(self.psr.freqs)),2])
		for i in range(len(np.unique(self.psr.freqs))):
			uniquezipped[i,0]=np.unique(self.psr.freqs)[i]
			uniquezipped[i,1]=np.sum(zipped[zipped[:,0]==np.unique(self.psr.freqs)[i]][:,1])

		zipped=uniquezipped

		totalweight=np.sum(weights)
		weightsum = np.cumsum(zipped[:,1])/totalweight

		if(FreqRange != None):
			chanindices = []
			for i in range(len(FreqRange)):
				if(len(chanindices) == 0):
					chanindices.append(FreqRange[i][0])
				if(len(chanindices) > 0 and FreqRange[i][0] != chanindices[len(chanindices)-1]):
					chanindices.append(FreqRange[i][0])
				chanindices.append(FreqRange[i][1])
		
	
		if(channels != None):
			chanindices = [minfreq-1]
			if(channels == len(uniquezipped)):
				for i in range(channels):	
					chanindices.append(zipped[i][0])
			else:
				for i in range(channels):
					newfreq=zipped[(np.abs(weightsum-np.float64(i+1)/channels)).argmin()][0]
					if(newfreq == chanindices[i]):
						print "this is the same :(", newfreq
					chanindices.append(zipped[(np.abs(weightsum-np.float64(i+1)/channels)).argmin()][0])

			chanindices[-1] += 1

			chanindices=np.array(chanindices)


		averageFreqs=[]
		for i in range(len(chanindices)-1):

			sub = zipped[np.logical_and(zipped[:,0] <= chanindices[i+1], zipped[:,0] > chanindices[i])]
			averageFreqs.append(np.sum(sub[:,0]*sub[:,1])/np.sum(sub[:,1]))

		averageFreqs=np.array(averageFreqs)
		self.TScrunchedFreqs = averageFreqs
		self.TScrunchChans = len(averageFreqs)

		TScrunchBins = []
		for i in range(self.TScrunchChans):
			TScrunchBins.append([0])

		for i in range(self.NToAs):


			freq = self.psr.freqs[i]
			Bins = self.Nbins[i]

			try:
				value,position = min((b,a) for a,b in enumerate(np.abs(self.TScrunchedFreqs-freq)) if b>=0)
			except:
				value,position = (maxfreq, self.TScrunchChans)

			if(TScrunchBins[position][0] == 0):
				TScrunchBins[position][0] = Bins
			else:
				if(Bins not in TScrunchBins[position]):
					TScrunchBins[position].append(Bins)

		self.TScrunchBins = TScrunchBins
		
					
		if(FromPickle == False):
	
			TScrunched = []
			for i in range(self.TScrunchChans):
				for j in range(len(TScrunchBins[i])):
					TScrunched.append(np.zeros(TScrunchBins[i][j]))
	


			totalweight = np.zeros(self.TScrunchChans)

			profcount = 0
			print "\nAveraging All Data In Time: "

                	RollBins = (np.floor(self.ShiftedBinTimes/(self.FoldingPeriods/self.Nbins[:])+0.5)).astype(np.int)	

			if(FromTM == False):
				while(profcount < self.NToAs):

					self.update_progress(np.float64(profcount)/self.NToAs)
					arch=psrchive.Archive_load(self.FNames[profcount])

			
					npol = arch.get_npol()
					if(npol>1):
						arch.pscrunch()

					arch.dedisperse()
					arch.centre()
					arch.remove_baseline()

					nsub=arch.get_nsubint()


					for i in range(nsub):

						if(profcount == self.NToAs):
							break

						subint=arch.get_Integration(i)

						nbins = subint.get_nbin()
						nchans = subint.get_nchan()
						npols = subint.get_npol()
						foldingperiod = subint.get_folding_period()
						inttime = subint.get_duration()
						centerfreq = subint.get_centre_frequency()


						firstbin = subint.get_epoch()
						intday = firstbin.intday()
						fracday = firstbin.fracday()
						intsec = firstbin.get_secs()
						fracsecs = firstbin.get_fracsec()
						isdedispersed = subint.get_dedispersed()

						pulsesamplerate = foldingperiod/nbins/self.SECDAY;

						nfreq=subint.get_nchan()

						FirstBinSec = intsec + np.float128(fracsecs)
						SubIntTimeDiff = FirstBinSec-self.SatSecs[profcount]*self.SECDAY
						PeriodDiff = SubIntTimeDiff*self.psr['F0'].val

						if(abs(PeriodDiff) < 2.0):
							for j in range(nfreq):
								chanfreq = subint.get_centre_frequency(j)
								toafreq = self.psr.freqs[profcount]
								prof=subint.get_Profile(0,j)
		

								if(FromTM == True):
									profamps = np.roll(self.ProfileData[profcount], RollBins[profcount])
								else:
									profamps = prof.get_amps()

								if(np.sum(profamps) != 0 and abs(toafreq-chanfreq) < 0.001):
									noiselevel=self.GetProfNoise(profamps)
									weight = 1.0/noiselevel**2
					
									try:
										value,position = min((b,a) for a,b in enumerate(chanindices-toafreq) if b>=0)
									except:
										value,position = (maxfreq, self.TScrunchChans)
				
									totalweight[position-1] += weight
									TScrunched[position-1] += profamps*weight

									profcount += 1
									if(profcount == self.NToAs):
										break

			if(FromTM == True):
				for i in range(self.NToAs):
					self.update_progress(np.float64(i)/self.NToAs)
					profamps = np.roll(self.ProfileData[i], RollBins[i])
					noiselevel=self.GetProfNoise(profamps)
                                        weight = 1.0/noiselevel**2

					toafreq = self.psr.freqs[i]

					try:
						value,position = min((b,a) for a,b in enumerate(chanindices-toafreq) if b>=0)
					except:
						value,position = (maxfreq, self.TScrunchChans)

					totalweight[position-1] += weight
					TScrunched[position-1] += profamps*weight

				#(TScrunched.T/totalweight).T

			for i in range(self.TScrunchChans):
				TScrunched[i] /= np.max(TScrunched[i])

			self.TScrunched = TScrunched
			self.TScrunchedNoise  = np.zeros(self.TScrunchChans)
			for i in range(self.TScrunchChans):
				self.TScrunchedNoise[i] = self.GetProfNoise(TScrunched[i])

			if(ToPickle == True):
				print "\nPickling TScrunch"
				output = open(self.root+'-TScrunch.'+str(self.TScrunchChans)+'C.pickle', 'wb')
				pickle.dump(self.TScrunched, output)
				pickle.dump(self.TScrunchedNoise, output)
				output.close()

		if(FromPickle == True):
			print "Loading TScrunch from Pickled Data"
			pick = open(self.root+'-TScrunch.'+str(self.TScrunchChans)+'C.pickle', 'rb')
			self.TScrunched = pickle.load(pick)
			self.TScrunchedNoise  = pickle.load(pick)
			pick.close()

		if(doplot == True):
			for i in range(self.TScrunchChans):
				plt.plot(np.linspace(0,1,len(self.TScrunched[i])), self.TScrunched[i])
				plt.xlabel('Phase')
				plt.ylabel('Channel '+str(i)+' Amp')
				plt.show()



	def FitEvoCoeffs(self, RFreq = 1400, polyorder = 1, doplot = False):

		self.EvoRefFreq = RFreq
		self.EvoNPoly = polyorder

		coeffs=self.MLShapeCoeff
		coeffs=np.array(coeffs).T

		Ncoeff=np.sum(self.MaxCoeff)

		RefCoeffs=np.zeros(Ncoeff)

		channels = self.TScrunchChans


		averageFreqs = self.TScrunchedFreqs
		refloc=np.abs(averageFreqs-RFreq).argmin()


		NewMLShapeCoeff = np.zeros([Ncoeff, polyorder+1])
		NewMLShapeCoeff[0][0] = 1
		for i in range(1,Ncoeff):

		#	coeffs[:][2][n::Ncoeff][refloc] = 10.0**-10
			line=np.zeros(len(coeffs[i]))
			c=np.polynomial.polynomial.polyfit((averageFreqs-RFreq)/1000, coeffs[i], polyorder, rcond=None, full=False, w=1.0/self.TScrunchShapeErr[i])
			NewMLShapeCoeff[i] = c
			#print i, c, coeffs[i]

			if(self.doplot == 2):
				for n in range(len(c)):
					line+=c[n]*((averageFreqs-RFreq)/1000)**n
				plt.errorbar((averageFreqs-RFreq)/1000, coeffs[i], yerr=self.TScrunchShapeErr[i],linestyle='')
				plt.errorbar((averageFreqs-RFreq)/1000, line)
				plt.show()

		self.MLShapeCoeff = NewMLShapeCoeff


	def my_prior(self, x):
	    logp = 0.

	    if np.all(x <= self.pmax) and np.all(x >= self.pmin):
		logp = np.sum(np.log(1/(self.pmax-self.pmin)))
	    else:
		logp = -np.inf

	    return logp



	def getInitialParams(self, MaxCoeff = 1, fitNComps = 1, RFreq = 1400, polyorder = 0, parameters = None, pmin = None, pmax = None, x0 = None, cov_diag = None, burnin = 1000, outDir = './Initchains/', sampler = 'pal', resume=False, incScattering = False, mn_live = 500, doplot=False):
	

		

		self.MaxCoeff = MaxCoeff
		self.fitNComps = fitNComps


		self.NScatterEpochs=0
		if(incScattering == True):
			self.NScatterEpochs=1

		if(parameters == None):
			parameters=[]
			for i in range(self.fitNComps):
				parameters.append('Phase_'+str(i))
			for i in range(self.fitNComps):
				parameters.append('Log10_Width_'+str(i))
			for i in range(self.fitNComps):
				parameters.append('NCoeff_'+str(i))

			if(incScattering == True):
				parameters.append('STau')

		print "\nGetting initial fit to profile using averaged data, fitting for: ", parameters
		n_params = len(parameters)


		if(pmin == None):
			pmin=[]
			for i in range(self.fitNComps):
				pmin.append(-0.5)
			for i in range(self.fitNComps):
				pmin.append(-3.5)
			for i in range(self.fitNComps):
				pmin.append(np.log10(1.0))
			if(incScattering == True):
				pmin.append(-6.0)


		if(pmax == None):
			pmax=[]
			for i in range(self.fitNComps):
				pmax.append(0.5)
			for i in range(self.fitNComps):
				pmax.append(0)
			for i in range(self.fitNComps):
				pmax.append(np.log10(MaxCoeff))
			if(incScattering == True):
				pmax.append(1.0)


		if(x0 == None):
			x0=[]
			for i in range(self.fitNComps):
				x0.append(0.0)
			for i in range(self.fitNComps):
				x0.append(-2)
			for i in range(self.fitNComps):
				x0.append(np.log10(50.0))
			if(incScattering == True):
				x0.append(-2.0)


		if(cov_diag == None):
			cov_diag=[]
			for i in range(self.fitNComps):
				cov_diag.append(0.1)
			for i in range(self.fitNComps):
				cov_diag.append(0.1)
			for i in range(self.fitNComps):
				cov_diag.append(0.1)
			if(incScattering == True):
				cov_diag.append(0.1)

		self.pmin = np.array(pmin)
		self.pmax = np.array(pmax)
		x0 = np.array(x0)
		cov_diag = np.array(cov_diag)


		self.doplot = 0
		self.returnVal = 0

		ML=[]

		if(sampler == 'pal'):
			sampler = ptmcmc.PTSampler(ndim=n_params,logl=self.InitialLogLike,logp=self.my_prior,
						    cov=np.diag(cov_diag**2),
						    outDir=outDir,
						    resume=resume)

			sampler.sample(p0=x0,Niter=10000,isave=10,burn=burnin,thin=1,neff=1000)

			chains=np.loadtxt(outDir+'/chain_1.txt').T

			self.chains = chains


			ML=chains.T[burnin:][np.argmax(chains[-3][burnin:])][:n_params]

		elif(sampler == 'multinest'):
                        
                        if not os.path.exists(outDir):
                            os.makedirs(outDir)

			pymultinest.run(self.MNFFTInitialLogLikeWrap, self.MNprior, n_params, importance_nested_sampling = False, resume = resume, verbose = True, sampling_efficiency = 'model', multimodal=False, n_live_points = mn_live, outputfiles_basename=outDir)

			chains=np.loadtxt(outDir+'phys_live.points').T
			ML=chains.T[np.argmax(chains[-2])][:n_params]


		self.doplot=doplot
		self.returnVal = 1

		CompSeps=[]
		Betas = []
		MCoeffs=[]
		for i in range(self.fitNComps):
			CompSeps.append(ML[i])
			Betas.append(10.0**ML[i + self.fitNComps])
			MCoeffs.append(np.floor(10.0**ML[i + 2*self.fitNComps]).astype(np.int))

		
		self.CompSeps = np.array(CompSeps)
		self.MaxCoeff = np.array(MCoeffs)
		self.MeanBeta = np.array(Betas)

		self.TotCoeff = np.sum(self.MaxCoeff)

		self.CompSeps[0] = 0
		
		if(incScattering == True):
			self.MeanScatter = ML[3*self.fitNComps]

		
		print "ML:", ML
		self.MLShapeCoeff, self.TScrunchShapeErr = self.FFTInitialLogLike(ML)
		


		self.TScrunchShapeErr = np.array(self.TScrunchShapeErr).T

		if(self.TScrunchChans > 1 and polyorder > 0):
			self.FitEvoCoeffs(RFreq, polyorder)

		if(polyorder == 0):
			newShape = np.zeros([np.sum(self.MaxCoeff), 2])
			newShape[:,0] = self.MLShapeCoeff[0]
			self.MLShapeCoeff = newShape

	def TNothpl(self, n,x, pl):


		a=2.0
		b=0.0
		c=1.0
		y0=1.0
		y1=2.0*x
		pl[0]=1.0
		pl[1]=2.0*x


		for k in range(2, n):

			c=2.0*(k-1.0);

			y0=y0/np.sqrt((k*1.0))
			y1=y1/np.sqrt((k*1.0))
			yn=(a*x+b)*y1-c*y0


			pl[k]=yn
			y0=y1
			y1=yn

	  
	def Bconst(self, width, i):
		return (1.0/np.sqrt(width))/np.sqrt(2.0**i*np.sqrt(np.pi))






	def MNprior(self, cube, ndim, nparams):
		for i in range(ndim):
		        cube[i] = (self.pmax[i]-  self.pmin[i])*cube[i] + self.pmin[i]
	


	def MNInitialLogLikeWrap(self, cube, ndim, nparams):

		x=np.zeros(ndim)
		for i in range(ndim):
			x[i] = cube[i]
		return self.InitialLogLike(x)


	def MNFFTInitialLogLikeWrap(self, cube, ndim, nparams):

		x=np.zeros(ndim)
		for i in range(ndim):
			x[i] = cube[i]
		return self.FFTInitialLogLike(x)
	
	def FFTInitialLogLike(self, x):

	    
		NComps = self.fitNComps


		pcount = 0
		phase = x[pcount:pcount+NComps]


		for i in range(1, NComps):
			phase[i] = phase[0] + phase[i]

		pcount += NComps

		width = 10.0**x[pcount:pcount+NComps]
		pcount += NComps

		FitNCoeff = np.floor(10.0**x[pcount:pcount+NComps]).astype(np.int)
		pcount += NComps

		if(self.NScatterEpochs == 1):
			STau = 10.0**x[pcount]
			pcount += 1
	    
	    
		loglike = 0
	    
	    
		'''Start by working out position in phase of the model arrival time'''

		ScrunchBins = self.TScrunchBins
		ScrunchChans = self.TScrunchChans
	    
		rfftfreqs  = []
		FullMatrix = []
		FullCompMatrix = []
		RealCompMatrix = []
		for i in range(ScrunchChans):
			rfftfreqs.append(np.linspace(0,0.5*ScrunchBins[i][0],0.5*ScrunchBins[i][0]+1))
			FullMatrix.append(np.ones([np.sum(FitNCoeff), len(2*np.pi*rfftfreqs[i])]))
			FullCompMatrix.append(np.zeros([np.sum(FitNCoeff), len(2*np.pi*rfftfreqs[i])]) + 0j)
			RealCompMatrix.append(np.zeros([np.sum(FitNCoeff), 2*len(2*np.pi*rfftfreqs[i])-2]))

	    
		for chan in range(ScrunchChans):
			ccount = 0
			for comp in range(NComps):
	    
				Beta = self.ReferencePeriod*width[comp]
				if(FitNCoeff[comp] > 1):
					self.TNothpl(FitNCoeff[comp], 2*np.pi*rfftfreqs[chan]*width[comp], FullMatrix[chan][ccount:ccount+FitNCoeff[comp]])
	    
				ExVec = np.exp(-0.5*(2*np.pi*rfftfreqs[chan]*width[comp])**2)
				FullMatrix[chan][ccount:ccount+FitNCoeff[comp]]=FullMatrix[chan][ccount:ccount+FitNCoeff[comp]]*ExVec*width[comp]

				for coeff in range(FitNCoeff[comp]):
					FullCompMatrix[chan][ccount+coeff] = FullMatrix[chan][ccount+coeff]*(1j**coeff)
	    
				rollVec = np.exp(2*np.pi*((phase[comp]+0.5)*ScrunchBins[chan][0])*rfftfreqs[chan]/ScrunchBins[chan][0]*1j)


				ScaleFactors = self.Bconst(width[comp]*self.ReferencePeriod, np.arange(FitNCoeff[comp]))




				FullCompMatrix[chan][ccount:ccount+FitNCoeff[comp]] *= rollVec
				for i in range(FitNCoeff[comp]):
					FullCompMatrix[chan][i+ccount] *= ScaleFactors[i]
	    
	    
				RealCompMatrix[chan][:,:len(2*np.pi*rfftfreqs[chan])-1] = np.real(FullCompMatrix[chan][:,1:len(2*np.pi*rfftfreqs[chan])])
				RealCompMatrix[chan][:,len(2*np.pi*rfftfreqs[chan])-1:] = -1*np.imag(FullCompMatrix[chan][:,1:len(2*np.pi*rfftfreqs[chan])])

				ccount+=FitNCoeff[comp]


		if(self.NScatterEpochs == 0):


			loglike = 0
			MLCoeff=[]
			MLErrs = []


			for i in range(self.TScrunchChans):

				MTM = np.dot(RealCompMatrix[i], RealCompMatrix[i].T)

				#Prior = 1000.0
				#diag=MTM.diagonal().copy()
				#diag += 1.0/Prior**2
				#np.fill_diagonal(MTM, diag)
				try:
					Chol_MTM = sp.linalg.cho_factor(MTM.copy())
				except:
					return -np.inf

				if(self.returnVal == 1):
					ShapeErrs = np.sqrt(np.linalg.inv(MTM.copy()).diagonal())



				FFTScrunched = np.fft.rfft(self.TScrunched[i])
				RealFFTScrunched = np.zeros(2*len(2*np.pi*rfftfreqs[i])-2)
				RealFFTScrunched[:len(2*np.pi*rfftfreqs[i])-1] = np.real(FFTScrunched[1:])
				RealFFTScrunched[len(2*np.pi*rfftfreqs[i])-1:] = np.imag(FFTScrunched[1:])

				Md = np.dot(RealCompMatrix[i], RealFFTScrunched)			
				ML = sp.linalg.cho_solve(Chol_MTM, Md)

				s = np.dot(RealCompMatrix[i].T, ML)

				r = RealFFTScrunched - s	
				noise  = self.TScrunchedNoise[i]*np.sqrt(ScrunchBins[i][0])/np.sqrt(2)

				loglike  += -0.5*np.sum(r**2)/noise**2  - 0.5*np.log(noise**2)*len(RealFFTScrunched)

				if(self.doplot == 1):

					bd = np.fft.rfft(self.TScrunched[i])
					bd[0] = 0 + 0j
					bdt = np.fft.irfft(bd)

					bm = np.zeros(len(np.fft.rfft(self.TScrunched[i]))) + 0j
					bm[1:] = s[:len(s)/2] + 1j*s[len(s)/2:]
					bmt = np.fft.irfft(bm)

					plt.plot(np.linspace(0,1,ScrunchBins[i][0]), bdt, color='black')
					plt.plot(np.linspace(0,1,ScrunchBins[i][0]), bmt, color='red')
					plt.xlabel('Phase')
					plt.ylabel('Profile Amplitude')
					plt.show()
					plt.plot(np.linspace(0,1,ScrunchBins[i][0]),bdt-bmt)
					plt.xlabel('Phase')
					plt.ylabel('Profile Residuals')
					plt.show()

			    	if(self.returnVal == 1):
					zml=ML[0]
					MLCoeff.append(ML[0:]/zml)
					MLErrs.append(ShapeErrs*self.TScrunchedNoise[i]/zml)

			if(self.returnVal == 1):
				return MLCoeff, MLErrs

		if(self.NScatterEpochs == 1):

			loglike = 0
			MLCoeff=[]
			MLErrs = []

			FullMatrix = FullCompMatrix
			for i in range(self.TScrunchChans):

				ScatterScale = (self.TScrunchedFreqs[i]*10.0**6)**4/10.0**(9.0*4.0)
				STime = STau/ScatterScale
				ScatterVec = self.ConvolveExp(np.linspace(0, ScrunchBins[i][0]/2, ScrunchBins[i][0]/2+1)/self.ReferencePeriod, STime, returnComp = True)

				ScatterMatrix = FullMatrix[i]*ScatterVec

				RealCompMatrix[i][:,:len(2*np.pi*rfftfreqs[i])-1] = np.real(ScatterMatrix[:,1:len(2*np.pi*rfftfreqs[i])])
				RealCompMatrix[i][:,len(2*np.pi*rfftfreqs[i])-1:] = -1*np.imag(ScatterMatrix[:,1:len(2*np.pi*rfftfreqs[i])])



				MTM = np.dot(RealCompMatrix[i], RealCompMatrix[i].T)

				#Prior = 1000.0
				#diag=MTM.diagonal().copy()
				#diag += 1.0/Prior**2
				#np.fill_diagonal(MTM, diag)
				try:
					Chol_MTM = sp.linalg.cho_factor(MTM.copy())
				except:
					return -np.inf

				if(self.returnVal == 1):
					ShapeErrs = np.sqrt(np.linalg.inv(MTM.copy()).diagonal())

				FFTScrunched = np.fft.rfft(self.TScrunched[i])
				RealFFTScrunched = np.zeros(2*len(2*np.pi*rfftfreqs[i])-2)
				RealFFTScrunched[:len(2*np.pi*rfftfreqs[i])-1] = np.real(FFTScrunched[1:])
				RealFFTScrunched[len(2*np.pi*rfftfreqs[i])-1:] = np.imag(FFTScrunched[1:])				

				Md = np.dot(RealCompMatrix[i], RealFFTScrunched)			
				ML = sp.linalg.cho_solve(Chol_MTM, Md)

				s = np.dot(RealCompMatrix[i].T, ML)

				r = RealFFTScrunched - s	

				#for bin in range(1024):
				#	print i, bin, self.TScrunched[i][bin], s[bin], self.TScrunchedNoise[i]


				noise = self.TScrunchedNoise[i]*np.sqrt(ScrunchBins[i][0])/np.sqrt(2)
				chanlike  = -0.5*np.sum(r**2)/noise**2 - 0.5*np.log(noise**2)*len(RealFFTScrunched)

				loglike += chanlike

				#print i, chanlike, self.TScrunchedNoise[i]
				if(self.doplot == 1):

				    bd = np.fft.rfft(self.TScrunched[i])
				    bd[0] = 0 + 0j
				    bdt = np.fft.irfft(bd)

				    bm = np.zeros(len(np.fft.rfft(self.TScrunched[i]))) + 0j
				    bm[1:] = s[:len(s)/2] + 1j*s[len(s)/2:]
				    bmt = np.fft.irfft(bm)

				    plt.plot(np.linspace(0,1,ScrunchBins[i][0]), bdt, color='black')
				    plt.plot(np.linspace(0,1,ScrunchBins[i][0]), bmt, color='red')
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Amplitude')
				    plt.show()
				    plt.plot(np.linspace(0,1,ScrunchBins[i][0]),bdt-bmt)
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Residuals')
				    plt.show()

				if(self.returnVal == 1):
				    zml=ML[0]
				    MLCoeff.append(ML/zml)
				    #print ShapeErrs, ML
				    MLErrs.append(ShapeErrs*self.TScrunchedNoise[i]/zml)


			if(self.returnVal == 1):
				return MLCoeff, MLErrs

		loglike -= self.TScrunchChans*np.sum(FitNCoeff)



		return loglike

	def PreComputeShapelets(self, interpTime = 1, MeanBeta = 0.1, ToPickle = False, FromPickle = False):


		print("Calculating Shapelet Interpolation Matrix : ", interpTime, MeanBeta);

		'''
		/////////////////////////////////////////////////////////////////////////////////////////////  
		/////////////////////////Profile Params//////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////
		'''
		#interpTime = 2
		#MeanBeta = copy.copy(self.MeanBeta)
		#ToPickle = False
		#FromPickle = False


		InterpBins = np.max(self.Nbins)

		numtointerpolate = np.int(self.ReferencePeriod/InterpBins/interpTime/10.0**-9)+1
		InterpolatedTime = self.ReferencePeriod/InterpBins/numtointerpolate
		self.InterpolatedTime  = InterpolatedTime	



		MeanBeta = MeanBeta*self.ReferencePeriod


		interpStep = self.ReferencePeriod/InterpBins/numtointerpolate


		self.InterpBasis = np.zeros([numtointerpolate, InterpBins, np.sum(self.MaxCoeff)])
		self.InterpJitterMatrix = np.zeros([numtointerpolate,InterpBins, np.sum(self.MaxCoeff)])

		for t in range(numtointerpolate):

			self.update_progress(np.float64(t)/numtointerpolate)



			samplerate = self.ReferencePeriod/InterpBins
			bintimes = np.linspace(0, samplerate*(InterpBins-1), InterpBins)

			ccount = 0
			for comp in range(self.fitNComps):

				binpos = t*interpStep + self.CompSeps[comp]*self.ReferencePeriod

				x = bintimes-binpos
				x = ( x + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2
				x =  x/MeanBeta[comp]
				 
				ExVec = np.exp(-0.5*(x)**2)


				hermiteMatrix = np.zeros([self.MaxCoeff[comp]+1,InterpBins])
				JitterMatrix = np.zeros([InterpBins,self.MaxCoeff[comp]])

				self.TNothpl(self.MaxCoeff[comp]+1, x, hermiteMatrix)

				hermiteMatrix *= ExVec

				ScaleFactors = self.Bconst(MeanBeta[comp], np.arange(self.MaxCoeff[comp]+1))
				for i in range(self.MaxCoeff[comp]+1):
					hermiteMatrix[i] *= ScaleFactors[i]


				hermiteMatrix = hermiteMatrix.T

				JitterMatrix[:,0] = (1.0/np.sqrt(2.0))*(-1.0*hermiteMatrix[:,1])/MeanBeta[comp]
				for i in range(1,self.MaxCoeff[comp]):
					JitterMatrix[:,i] = (1.0/np.sqrt(2.0))*(np.sqrt(1.0*i)*hermiteMatrix[:,i-1] - np.sqrt(1.0*(i+1))*hermiteMatrix[:,i+1])/MeanBeta



				self.InterpBasis[t][:,ccount:ccount+self.MaxCoeff[comp]]  = np.copy(hermiteMatrix[:,:self.MaxCoeff[comp]])
				self.InterpJitterMatrix[t][:, ccount:ccount+self.MaxCoeff[comp]] = np.copy(JitterMatrix)

				ccount += self.MaxCoeff[comp]


	def PreComputeFFTShapelets(self, interpTime = 1, MeanBeta = 0.1, ToPickle = False, FromPickle = False, doplot = False, useNFBasis = 0):


		print("Calculating Shapelet Interpolation Matrix : ", interpTime, MeanBeta);

		'''
		/////////////////////////////////////////////////////////////////////////////////////////////  
		/////////////////////////Profile Params//////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////
		'''

		InterpBins = np.max(self.Nbins)
		MinBins    = np.min(self.Nbins)
		MaxBins    = np.max(self.Nbins)
		BinRatio   = MaxBins/MinBins

		numtointerpolate = np.int(BinRatio*self.ReferencePeriod/InterpBins/interpTime/10.0**-9)+1
		InterpolatedTime = self.ReferencePeriod/InterpBins/(numtointerpolate/BinRatio)
		self.InterpolatedTime  = InterpolatedTime	
		self.BinRatio = BinRatio


		lenRFFT = len(np.fft.rfft(np.ones(InterpBins)))

		interpStep = 1.0/InterpBins/(numtointerpolate/BinRatio)


		if(FromPickle == False):


			InterpFShapeMatrix = np.zeros([numtointerpolate, lenRFFT, np.sum(self.MaxCoeff)])+0j
			InterpFJitterMatrix = np.zeros([numtointerpolate,lenRFFT, np.sum(self.MaxCoeff)])+0j


			for t in range(numtointerpolate):

				self.update_progress(np.float64(t)/numtointerpolate)

				#binpos = t*interpStep

				rfftfreqs=np.linspace(0,0.5*InterpBins,0.5*InterpBins+1)

				ccount = 0
				for comp in range(self.fitNComps):

					binpos = -t*interpStep - self.CompSeps[comp]
					binpos = ( binpos + 0.5) % 1 - 0.5

					OneMatrix = np.ones([self.MaxCoeff[comp]+1, len(2*np.pi*rfftfreqs)])
					OneCompMatrix = np.zeros([self.MaxCoeff[comp]+1, len(2*np.pi*rfftfreqs)]) + 0j
					OneJitterMatrix = np.zeros([self.MaxCoeff[comp]+1, len(2*np.pi*rfftfreqs)]) + 0j

					if(self.MaxCoeff[comp]+1 > 1):
						self.TNothpl(self.MaxCoeff[comp]+1, 2*np.pi*rfftfreqs*MeanBeta[comp], OneMatrix)

					ExVec = np.exp(-0.5*(2*np.pi*rfftfreqs*MeanBeta[comp])**2)
					OneMatrix = OneMatrix*ExVec*InterpBins*np.sqrt(2*np.pi*MeanBeta[comp]**2)

					for coeff in range(self.MaxCoeff[comp]+1):
						OneCompMatrix[coeff] = OneMatrix[coeff]*(1j**coeff)

					#rollVec = np.exp(-2*np.pi*((-binpos)*InterpBins)*rfftfreqs/InterpBins*1j)
					rollVec = np.exp(-2*np.pi*((binpos)*InterpBins)*rfftfreqs/InterpBins*1j)


					ScaleFactors = self.Bconst(MeanBeta[comp]*self.ReferencePeriod, np.arange(self.MaxCoeff[comp]+1))




					OneCompMatrix *= rollVec
					for i in range(self.MaxCoeff[comp]+1):
						OneCompMatrix[i] *= ScaleFactors[i]

					OneCompMatrix = np.conj(OneCompMatrix)

					OneJitterMatrix[0] = (1.0/np.sqrt(2.0))*(-1.0*OneCompMatrix[1])/(MeanBeta[comp]*self.ReferencePeriod)
					for i in range(1,self.MaxCoeff[comp]):
						OneJitterMatrix[i] = (1.0/np.sqrt(2.0))*(np.sqrt(1.0*i)*OneCompMatrix[i-1] - np.sqrt(1.0*(i+1))*OneCompMatrix[i+1])/(MeanBeta[comp]*self.ReferencePeriod)


					OneCompMatrix = OneCompMatrix.T
					OneJitterMatrix = OneJitterMatrix.T

					InterpFShapeMatrix[t][:,ccount:ccount+self.MaxCoeff[comp]]  = np.copy(OneCompMatrix[:,:self.MaxCoeff[comp]])
					InterpFJitterMatrix[t][:,ccount:ccount+self.MaxCoeff[comp]] = np.copy(OneJitterMatrix[:,:self.MaxCoeff[comp]])


					ccount+=self.MaxCoeff[comp]




			threshold = 10.0**-10
			upperindex=1
			while(np.max(np.abs(InterpFShapeMatrix[0,upperindex:,:])) > threshold):
				upperindex += 5
				if(upperindex >= lenRFFT):
					upperindex = lenRFFT-1
					break
				print "upper index is:", upperindex,np.max(np.abs(InterpFShapeMatrix[0,upperindex:,:]))
			#InterpShapeMatrix = np.array(InterpShapeMatrix)
			#InterpJitterMatrix = np.array(InterpJitterMatrix)
			print("\nFinished Computing Interpolated Profiles")

			if(useNFBasis > 0):
				self.NFBasis = useNFBasis
			else:
				self.NFBasis = upperindex - 1

			self.InterpFBasis = np.zeros([numtointerpolate, self.NFBasis*2, self.TotCoeff])
			self.InterpJBasis = np.zeros([numtointerpolate, self.NFBasis*2, self.TotCoeff])

			self.InterpFBasis[:,:self.NFBasis,:] = InterpFShapeMatrix[:,1:self.NFBasis+1].real
			self.InterpFBasis[:,self.NFBasis:,:] = InterpFShapeMatrix[:,1:self.NFBasis+1].imag

			self.InterpJBasis[:,:self.NFBasis,:] = InterpFJitterMatrix[:,1:self.NFBasis+1].real
			self.InterpJBasis[:,self.NFBasis:,:] = InterpFJitterMatrix[:,1:self.NFBasis+1].imag

			#self.InterpFBasis = InterpFShapeMatrix[:,1:upperindex]
			#self.InterpFJitterMatrix = InterpFJitterMatrix[:,1:upperindex]

			self.InterpolatedTime  = InterpolatedTime

			self.ProfileFData = np.zeros([self.NToAs, 2*self.NFBasis])
			for i in range(self.NToAs):
				Fdata =  np.fft.rfft(self.ProfileData[i])[1:self.NFBasis+1]
				self.ProfileFData[i, :self.NFBasis] = np.real(Fdata)
				self.ProfileFData[i, self.NFBasis:] = np.imag(Fdata)

			if(ToPickle == True):
				print "\nPickling Basis"
				output = open(self.root+'-TScrunch.Basis.pickle', 'wb')
				pickle.dump(self.ProfileFData, output)
				pickle.dump(self.InterpFJitterMatrix, output)
				pickle.dump(self.InterpFBasis, output)
				output.close()

		if(FromPickle == True):
			print "Loading Basis from Pickled Data"
			pick = open(self.root+'-TScrunch.Basis.pickle', 'rb')
			self.ProfileFData = pickle.load(pick)
			self.InterpFJitterMatrix  = pickle.load(pick)
			self.InterpFBasis = pickle.load(pick)
			pick.close()
			self.NFBasis = np.shape(self.InterpFBasis)[1]
			print "Loaded NFBasis: ", self.NFBasis

		if(doplot == True):

		   
			bm = np.zeros(len(np.fft.rfft(np.zeros(InterpBins)))) + 0j
			sr = np.dot(self.InterpFBasis[0][:self.SavedNFBasis], self.MLShapeCoeff[:,0]) 
			si = np.dot(self.InterpFBasis[0][self.SavedNFBasis:], self.MLShapeCoeff[:,0])
			bm[1:self.SavedNFBasis+1] = sr + 1j*si
			bmfft = np.fft.irfft(bm)

			plt.plot(np.linspace(0,1,InterpBins), bmfft)

			plt.xlabel('Phase')
			plt.ylabel('Profile Amplitude')
			plt.show()






	def getInitialPhase(self, doplot = True, ToPickle = False, FromPickle = False):
	

		if(FromPickle == False):
			print "Getting initial fit to phase using full data"


			self.doplot = False

			phases=[]
			likes=[]
			minphase = 0.0
			maxphase = 1.0
			for loop in range(4):
				stepsize = (maxphase-minphase)/100
				mlike = -np.inf
				mphase=0
				for p in range(100):
					self.update_progress(np.float64(loop*100 + p)/400.0)
					phase = minphase + p*stepsize
					like = self.FFTPhaseLike(np.ones(1)*phase)
					phases.append(phase)
					likes.append(like)
					if(like > mlike):
						mlike = like
						mphase = np.ones(1)*phase
						#print mphase, mlike

		
				minphase = mphase - stepsize
				maxphase = mphase + stepsize

			if(doplot == True):
				plt.scatter(phases, likes)
				plt.xlabel('Phase')
				plt.ylabel('Log-Likelihood')
				plt.show()

			self.MeanPhase = mphase

			if(ToPickle == True):
                                print "Saving Data"
                                output = open(self.root+'-Phase.pickle', 'wb')
                                pickle.dump(self.MeanPhase, output)
                                output.close()

                if(FromPickle == True):
                        print "Loading from Saved Data"
                        pick = open(self.root+'-Phase.pickle', 'rb')
                        self.MeanPhase = pickle.load(pick)
                        pick.close()	

		print "Using Mean Phase: ", self.MeanPhase
		print "\n"

		#self.getShapeletStepSize = True
		#self.hess = self.PhaseLike(np.ones(1)*mphase)
		#self.getShapeletStepSize = False

		#self.doplot=True
		#self.PhaseLike(ML)
		


	def FFTPhaseLike(self, x):

		phase=x[0]*self.ReferencePeriod
		NCoeff = self.MaxCoeff-1

		ShapeAmps=self.MLShapeCoeff

		loglike = 0


		OneBin=self.FoldingPeriods/self.Nbins
		InterpSize = np.shape(self.InterpFBasis)[0]


		xS = self.ShiftedBinTimes-phase
		xS = ( xS + self.FoldingPeriods/2) % (self.FoldingPeriods ) - self.FoldingPeriods/2

		InterpBins = (np.floor(-xS%(OneBin)/self.InterpolatedTime+0.5)).astype(int)%InterpSize
		WBTs = xS+self.InterpolatedTime*(InterpBins-1)
		RollBins=(np.floor(WBTs/OneBin+0.5)).astype(np.int)


		for i in range(self.NToAs):

			s = np.sum([np.dot(self.InterpFBasis[InterpBins[i]], ShapeAmps[:,c])*(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**c) for c in range(self.EvoNPoly+1)], axis=0)

			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

			pnoise = self.ProfileInfo[i,6]*np.sqrt(self.Nbins[i])/np.sqrt(2)

			RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
			ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RealRoll*self.ProfileFData[i][:self.NFBasis]-ImagRoll*self.ProfileFData[i][self.NFBasis:]
			RollData[self.NFBasis:] = ImagRoll*self.ProfileFData[i][:self.NFBasis]+RealRoll*self.ProfileFData[i][self.NFBasis:]



			if(self.NScatterEpochs > 0):
				ScatterScale = self.SSBFreqs[i]**4/10.0**(9.0*4.0)
				STime = (10.0**self.MeanScatter)/ScatterScale
				ScatterVec = self.ConvolveExp(rfftfreqs*self.Nbins[i]/self.FoldingPeriods[i], STime, returnComp = True)

				RealScattered = ScatterVec.real*s[:self.NFBasis]+ScatterVec.imag*s[self.NFBasis:]
				ImagScattered =  -ScatterVec.imag*s[:self.NFBasis]+ScatterVec.real*s[self.NFBasis:]

				s[:self.NFBasis] = RealScattered
				s[self.NFBasis:] = ImagScattered

			FS = np.zeros(2*self.NFBasis)
			FS[:self.NFBasis] = s[:self.NFBasis]
			FS[self.NFBasis:] = s[self.NFBasis:]

			FS /= np.sqrt(np.dot(FS,FS)/(2*self.NFBasis))

			MNM = np.dot(FS, FS)/(pnoise*pnoise)
			detMNM = MNM
			logdetMNM = np.log(detMNM)

			InvMNM = 1.0/MNM

			dNM = np.dot(RollData, FS)/(pnoise*pnoise)
			dNMMNM = dNM*InvMNM

			MarginLike = dNMMNM*dNM

			profilelike = -0.5*(logdetMNM - MarginLike)
			loglike += profilelike     

			if(self.doplot == True):

				plt.plot(np.linspace(0,2,2*self.NFBasis), RollData, color='black')
				plt.plot(np.linspace(0,2,2*self.NFBasis), dNMMNM*FS, color='red')
				plt.xlabel('Frequency')
				plt.ylabel('Profile Amplitude')
				plt.show()


				bd = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
				bd[1:self.NFBasis+1] = RollData[:self.NFBasis] + 1j*RollData[self.NFBasis:]
				bdt = np.fft.irfft(bd)

				bm = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
				bm[1:self.NFBasis+1] = dNMMNM*FS[:self.NFBasis] + 1j*dNMMNM*FS[self.NFBasis:]
				bmt = np.fft.irfft(bm)

				plt.plot(np.linspace(0,1,self.Nbins[i]), bdt, color='black')
				plt.plot(np.linspace(0,1,self.Nbins[i]), bmt, color='red')
				plt.xlabel('Phase')
				plt.ylabel('Profile Amplitude')
				plt.show()
				plt.plot(np.linspace(0,1,self.Nbins[i]),bdt-bmt)
				plt.xlabel('Phase')
				plt.ylabel('Profile Residuals')
				plt.show()


		return loglike




	def calculateGHSHessian(self, diagonalGHS = False):

		NCoeff = self.TotCoeff-1

		x0 = np.zeros(self.n_params)
		cov_diag = np.zeros(self.n_params)

		DenseParams = self.DenseParams
		hess_dense = np.zeros([DenseParams,DenseParams])

		if(self.incBaselineNoise == True):
			self.BLNHess = np.zeros([self.NToAs, self.BaselineNoiseParams, self.BaselineNoiseParams])
			self.BLNEigM = np.zeros([self.NToAs, self.BaselineNoiseParams, self.BaselineNoiseParams])

		LinearSize = self.LinearParams

		#####################Get Parameters####################################

		if(self.incPAmps == True):
			ProfileAmps = self.MLParameters[self.ParamDict['PAmps'][2]]

		if(self.incPNoise == True):
			ProfileNoise = self.MLParameters[self.ParamDict['PNoise'][2]]


		if(self.incPhase == True):
			Phase = self.MLParameters[self.ParamDict['Phase'][2]][0]
		else:
			Phase = 0

		if(self.incLinearTM == True):
			TimingParameters = self.MLParameters[self.ParamDict['LinearTM'][2]]
			TimeSignal = np.dot(self.designMatrix, TimingParameters)


		if(self.incProfile == True):
			ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
			ShapeAmps[0][0] = 1
			ShapeAmps[1:]=self.MLParameters[self.ParamDict['Profile'][2]].reshape([NCoeff,(self.EvoNPoly+1)])
	
		JitterSignal = np.zeros(self.NToAs)

		if(self.incEQUAD == True):
			EQUADSignal = self.MLParameters[self.ParamDict['EQUADSignal'][2]]
			EQUADPriors  = 10.0**self.MLParameters[self.ParamDict['EQUADPrior'][2]]
	
			for i in range(self.NumEQPriors):
				EQIndicies = np.where(self.EQUADInfo==i)[0]
				Prior = EQUADPriors[i]
				if(self.EQUADModel[i] == -1 or self.EQUADModel[i] == 0):
					EQUADSignal[EQIndicies] *= Prior
			JitterSignal += EQUADSignal

		if(self.incECORR == True):
			ECORRSignal = copy.copy(self.MLParameters[self.ParamDict['ECORRSignal'][2]])
			ECORRPriors  = 10.0**self.MLParameters[self.ParamDict['ECORRPrior'][2]]
	
			for i in range(self.NumECORRPriors):
				ECORRIndicies = np.where(self.ECORRInfo==i)[0]
				Prior = ECORRPriors[i]
				if(self.ECORRModel[i] == -1 or self.ECORRModel[i] == 0):
					ECORRSignal[ECORRIndicies] *= Prior
	
			JitterSignal += ECORRSignal[self.EpochIndex]

		if(self.incScatter == True):
			ScatterFreqScale = self.MLParameters[self.ParamDict['ScatterFreqScale'][2]]
			ScatteringParameters = 10.0**self.MLParameters[self.ParamDict['Scattering'][2]]

		if(self.incBaselineNoise == True):

			BaselineNoisePriorAmps  = copy.copy(self.MLParameters[self.ParamDict['BaselineNoiseAmpPrior'][2]])
			BaselineNoisePriorSpecs  = copy.copy(self.MLParameters[self.ParamDict['BaselineNoiseSpecPrior'][2]])


		xS = self.ShiftedBinTimes-Phase*self.ReferencePeriod-JitterSignal

		if(self.incLinearTM == True):
			xS -= TimeSignal


		OneBin=self.FoldingPeriods/self.Nbins
		InterpSize = np.shape(self.InterpFBasis)[0]


	
		xS = ( xS + self.FoldingPeriods/2) % (self.FoldingPeriods ) - self.FoldingPeriods/2

		InterpBins = (np.floor(-xS%(OneBin)/self.InterpolatedTime+0.5)).astype(int)%InterpSize
		WBTs = xS+self.InterpolatedTime*(InterpBins-1)
		RollBins=(np.floor(WBTs/OneBin+0.5)).astype(np.int)

		OneFBasis = self.InterpFBasis[InterpBins]
		OneJBasis = self.InterpJBasis[InterpBins]

		ssbfreqs = self.psr.ssbfreqs()/10.0**6

		#ProfAmps = ShapeAmps[:,0] +  ShapeAmps[:,1]*(((ssbfreqs[0] - self.EvoRefFreq)/1000.0))

		s = np.sum([np.dot(OneFBasis, ShapeAmps[:,i])*(((self.psr.ssbfreqs()/10.0**6 - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)

		j = np.sum([np.dot(OneJBasis, ShapeAmps[:,i])*(((self.psr.ssbfreqs()/10.0**6 - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)

		like = 0
		chisq = 0
		detN = 0
		for i in range(self.NToAs):

			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

			RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
			ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)


			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RealRoll*self.ProfileFData[i][:self.NFBasis]-ImagRoll*self.ProfileFData[i][self.NFBasis:]
			RollData[self.NFBasis:] = ImagRoll*self.ProfileFData[i][:self.NFBasis]+RealRoll*self.ProfileFData[i][self.NFBasis:]

			if(self.NScatterEpochs > 0):

				NoScatterS = copy.copy(s[i])

				tau = np.sum(ScatteringParameters[self.ScatterInfo[i]])
				f = np.linspace(1,self.NFBasis,self.NFBasis)/self.FoldingPeriods[i]
				w = 2.0*np.pi*f
				ISS = 1.0/(self.psr.ssbfreqs()[i]**ScatterFreqScale/self.ScatterRefFreq**(ScatterFreqScale))
				ISS2 = 1.0/(self.psr.ssbfreqs()[i]**ScatterFreqScale/10.0**(9.0*ScatterFreqScale))
				#ISS = 1.0/((self.psr.ssbfreqs()[i]**ScatterFreqScale)/(self.ScatterRefFreq**(ScatterFreqScale)))
				#print i, self.psr.freqs[i], ISS, ISS2, tau*ISS
				RConv, IConv = self.ConvolveExp(f, tau*ISS)

				RConfProf = RConv*s[i][:self.NFBasis] - IConv*s[i][self.NFBasis:]
				IConfProf = IConv*s[i][:self.NFBasis] + RConv*s[i][self.NFBasis:]

				s[i][:self.NFBasis] = RConfProf
				s[i][self.NFBasis:] = IConfProf

				RConfProf = RConv*j[i][:self.NFBasis] - IConv*j[i][self.NFBasis:]
				IConfProf = IConv*j[i][:self.NFBasis] + RConv*j[i][self.NFBasis:]

				j[i][:self.NFBasis] = RConfProf
				j[i][self.NFBasis:] = IConfProf

				RBasis = (RConv*OneFBasis[i,:self.NFBasis,:].T - IConv*OneFBasis[i,self.NFBasis:,:].T).T
				IBasis = (IConv*OneFBasis[i,:self.NFBasis,:].T + RConv*OneFBasis[i,self.NFBasis:,:].T).T 

				OneFBasis[i,:self.NFBasis,:] = RBasis
				OneFBasis[i,self.NFBasis:,:] = IBasis



			MNM = np.dot(s[i], s[i])
			dNM = np.dot(RollData, s[i])

			if(ProfileAmps[i] == None):
				MLAmp = dNM/MNM
				self.MLParameters[self.ParamDict['PAmps'][2]][i] = MLAmp
			else:
				MLAmp = ProfileAmps[i]

			PSignal = MLAmp*s[i]
			'''
			if(i==0 or i == 100 or i == 200 or i == 300 or i == 400 or i == 500 or i == 600):
				plt.plot(np.linspace(0,2,2*self.NFBasis), RollData, color='black')
				plt.plot(np.linspace(0,2,2*self.NFBasis), PSignal, color='red')
				plt.xlabel('Frequency')
				plt.ylabel('Profile Amplitude')
				plt.show()

				plt.plot(np.linspace(0,2,2*self.NFBasis), RollData-PSignal, color='black')
				plt.xlabel('Frequency')
				plt.ylabel('Profile Amplitude')
				plt.show()

				bd = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
				bd[1:self.NFBasis+1] = RollData[:self.NFBasis] + 1j*RollData[self.NFBasis:]
				bdt = np.fft.irfft(bd)

				bm = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
				bm[1:self.NFBasis+1] = PSignal[:self.NFBasis] + 1j*PSignal[self.NFBasis:]
				bmt = np.fft.irfft(bm)

				plt.plot(np.linspace(0,1,self.Nbins[i]), bdt, color='black')
				plt.plot(np.linspace(0,1,self.Nbins[i]), bmt, color='red')
				plt.xlabel('Phase')
				plt.ylabel('Profile Amplitude')
				plt.show()
				plt.plot(np.linspace(0,1,self.Nbins[i]),bdt-bmt)
				plt.xlabel('Phase')
				plt.ylabel('Profile Residuals')
				plt.show()
			'''



			Res=RollData-PSignal

			RR = np.dot(Res, Res)

			if(ProfileNoise[i] == None):		
				MLSigma =  np.std(Res)
				self.MLParameters[self.ParamDict['PNoise'][2]][i] = MLSigma
			else:
				MLSigma = ProfileNoise[i]

			Noise = np.ones(2*self.NFBasis)*MLSigma**2
	

			if(self.incBaselineNoise == True):

                                BLRefF = self.BaselineNoiseRefFreq
                                BLNFreqs = np.zeros(2*self.NFBasis)
                                BLNFreqs[:self.NFBasis] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)
                                BLNFreqs[self.NFBasis:] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)

                                Amp=10.0**(2*BaselineNoisePriorAmps[i])
                                Spec = BaselineNoisePriorSpecs[i]
                                BLNPower = Amp*pow(BLNFreqs, -Spec)

				BLNPower[self.NFBasis-5:self.NFBasis] = 0
				BLNPower[-5:] = 0

				Noise += BLNPower

			
			MNM = np.dot(s[i], s[i]/Noise)
			dNM = np.dot(RollData, s[i]/Noise)

			if(ProfileAmps[i] == None):
				MLAmp = dNM/MNM
				self.MLParameters[self.ParamDict['PAmps'][2]][i] = MLAmp


			like += 0.5*np.sum(Res*Res/Noise) + 0.5*np.sum(np.log(Noise))
			chisq += 0.5*np.sum(Res*Res/Noise)
			detN += 0.5*np.sum(np.log(Noise))

			#print i, MLAmp, MLSigma, chisq, detN, like

			if(self.fitPAmps == True):
				index=self.ParamDict['PAmps'][0]+i
				x0[index] = MLAmp
				cov_diag[index] = MNM


			if(self.fitPNoise == True):
				index=self.ParamDict['PNoise'][0]+i
				x0[index] = MLSigma
				cov_diag[index] = np.sum(-2*MLSigma**2/Noise**2 + 1.0/Noise + 4*MLSigma**2*Res**2/Noise**3 - Res**2/Noise**2)
				#3*RR/(MLSigma*MLSigma*MLSigma*MLSigma) - 2.0*self.NFBasis/(MLSigma*MLSigma)


			BLNIndex = 0
			if(self.fitBaselineNoiseAmpPrior == True):
			
				Top = np.log(10)*BLNPower
				T1 = -2*Top**2/Noise**2 
				T2 = 2*Top*np.log(10.0)/Noise
				T3 = 4*Top**2*Res**2/Noise**3
				T4 = -2*Top*np.log(10.0)*Res**2/Noise**2
			
				HTerm = np.sum(T1 + T2 + T3 + T4)
	
				if(HTerm < 0):
					HTerm = 5.0
				
				#print i, HTerm
				index=self.ParamDict['BaselineNoiseAmpPrior'][0]+i
				x0[index] = -1
				cov_diag[index] = HTerm
				self.BLNHess[i,BLNIndex, BLNIndex]  =  cov_diag[index]

				BLNIndex += 1


			if(self.fitBaselineNoiseSpecPrior == True):
		
                                Top = np.log(BLNFreqs)*BLNPower
                                T1 = -0.5*Top**2/Noise**2 
                                T2 = 0.5*Top*np.log(BLNFreqs)/Noise
                                T3 = Top**2*Res**2/Noise**3
                                T4 = -0.5*Top*np.log(BLNFreqs)*Res**2/Noise**2

                                HTerm = np.sum(T1 + T2 + T3 + T4)

                                if(HTerm < 0):
                                        HTerm = 5.0

				index=self.ParamDict['BaselineNoiseSpecPrior'][0]+i
				x0[index] = 4
				cov_diag[index] = HTerm



				self.BLNHess[i,BLNIndex, BLNIndex]  = cov_diag[index]

				if(self.fitBaselineNoiseAmpPrior == True):
					
	                                Top = BLNPower
					T1 =  Top**2*np.log(10.0)*np.log(BLNFreqs)/Noise**2
					T2 = -Top*np.log(10.0)*np.log(BLNFreqs)/Noise
					T3 = -2*Top**2*Res**2*np.log(10.0)*np.log(BLNFreqs)/Noise**3
					T4 =  Top*np.log(10.0)*np.log(BLNFreqs)*Res**2/Noise**2
					
					self.BLNHess[i, -1, -2]  =  np.sum(T1 + T2 + T3 + T4)
					self.BLNHess[i, -2, -1]  =  np.sum(T1 + T2 + T3 + T4)

					


				

			#Make Matrix for Linear Parameters

			HessMatrix = np.zeros([LinearSize, 2*self.NFBasis])


			PhaseScale = -1*MLAmp/np.sqrt(Noise)
			LinCount = 0

			for key in self.ParamDict.keys():
				if(self.ParamDict[key][5] == 1):

					#Hessian for Profile Amp (if dense)
					if(key == 'PAmps' and self.fitPAmps == True and self.DensePAmps == True):
						HessMatrix[LinCount,:] = s[i]/np.sqrt(Noise)
						LinCount += 1

					if(key == 'EQUADSignal'and self.fitEQUADSignal == True):
						EQIndex = np.int(self.EQUADInfo[i])

						if(self.EQUADModel[EQIndex] == -1):
							HessMatrix[LinCount, :] = 0				

						if(self.EQUADModel[EQIndex] == 0):
							HessMatrix[LinCount, :] = j[i]*PhaseScale*EQUADPriors[EQIndex]

						if(self.EQUADModel[EQIndex] == 1):
							HessMatrix[LinCount, :] = j[i]*PhaseScale

						LinCount += 1
	
					if(key == 'ECORRSignal'and self.fitECORRSignal == True):
						EpochIndex = self.EpochIndex[i]
						ECORRIndex = np.int(self.ECORRInfo[EpochIndex])

						if(self.ECORRModel[ECORRIndex] == -1):
							HessMatrix[LinCount, :] = 0				

						if(self.ECORRModel[ECORRIndex] == 0):
							HessMatrix[LinCount, :] = j[i]*PhaseScale*ECORRPriors[ECORRIndex]

						if(self.ECORRModel[ECORRIndex] == 1):
							HessMatrix[LinCount, :] = j[i]*PhaseScale

						LinCount += 1


					#Hessian for Phase parameter
					if(key == 'Phase' and self.fitPhase == True):
						HessMatrix[LinCount,:] = PhaseScale*j[i]*self.FoldingPeriods[i]
						LinCount += 1

					#Hessian for Timing Model
					if(key == 'LinearTM' and self.fitLinearTM == True):
						for c in range(self.numTime):
							HessMatrix[LinCount, :] = j[i]*PhaseScale*self.designMatrix[i,c]
							LinCount += 1


					#Hessian for Shapelet parameters
					if(key == 'Profile' and self.fitProfile == True):
						fvals = ((self.psr.ssbfreqs()[i]/10.0**6 - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)

						ShapeBasis = OneFBasis[i]

						for c in range(1, self.TotCoeff):
							for p in range(self.EvoNPoly+1):
								HessMatrix[LinCount, :] = fvals[p]*ShapeBasis[:,c]*MLAmp/np.sqrt(Noise)
								LinCount += 1



			OneHess = np.dot(HessMatrix, HessMatrix.T)
			DiagHess = OneHess.diagonal()


			######################Now Copy elements from OneHess to the full Hessian##############################

			LinCount1 = 0
			for k1 in range(len(self.ParamDict.keys())):
				key1 = self.ParamDict.keys()[k1]
				if(self.ParamDict[key1][5] == 1):
					LinCount2 = LinCount1
					Np1 = self.ParamDict[key1][1] - self.ParamDict[key1][0]
					index1 = self.ParamDict[key1][0] - self.DiagParams

					if(key1 == 'PAmps' or key1 == 'EQUADSignal'): 
						index1 += i
						Np1 = 1
	
					if(key1 == 'ECORRSignal'):
						index1 += self.EpochIndex[i]
						Np1 = 1
	
					#print "param1: ", key1, Np1, index1
					hess_dense[index1:index1+Np1, index1:index1+Np1] += OneHess[LinCount1:LinCount1+Np1, LinCount1:LinCount1+Np1]
					cov_diag[index1+self.DiagParams:index1+Np1+self.DiagParams] +=  DiagHess[LinCount1:LinCount1+Np1]
					LinCount2 += Np1

					for k2 in range(k1+1, len(self.ParamDict.keys())):
						key2 = self.ParamDict.keys()[k2]
						if(self.ParamDict[key2][5] == 1):
							Np2 = self.ParamDict[key2][1] - self.ParamDict[key2][0]
							index2 = self.ParamDict[key2][0] - self.DiagParams

							if(key2 == 'PAmps' or key2 == 'EQUADSignal'): 
								index2 += i
								Np2 = 1
			
							
							if(key2 == 'ECORRSignal'):
								index2 += self.EpochIndex[i]
								Np2= 1

							#print "param2: ", key2, Np2, LinCount1, LinCount2, index1, index2, Np1
							hess_dense[index1: index1+Np1, index2: index2+Np2] += OneHess[LinCount1: LinCount1+Np1,  LinCount2: LinCount2+Np2]
							hess_dense[index2: index2+Np2, index1: index1+Np1] += OneHess[LinCount2: LinCount2+Np2,  LinCount1: LinCount1+Np1]

							LinCount2 += Np2

					LinCount1 += Np1

			######################Add any priors to the hessian######################



			if(self.fitPhase == True):

				index = self.ParamDict['Phase'][0] - self.DiagParams
				hess_dense[index,index]  += (1.0/self.PhasePrior/self.PhasePrior)/self.NToAs
				#print index

			if(self.fitEQUADSignal == True):
				EQIndex = np.int(self.EQUADInfo[i])
				index = self.ParamDict['EQUADSignal'][0] + i - self.DiagParams
				#print index

				if(self.EQUADModel[EQIndex] == -1):
					#print "adding prior", -1, 1
					hess_dense[index,index]  += 1.0

				if(self.EQUADModel[EQIndex] == 0):
					#print "adding prior", 0, 1 
					hess_dense[index,index]  += 1.0
					#Prior = 1.0/EQUADPriors[EQIndex]/EQUADPriors[EQIndex]
					#hess_dense[index,index]  += Prior#/np.sum(self.EQUADInfo==self.EQUADInfo[i])

				if(self.EQUADModel[EQIndex] == 1):
					#print "adding prior", 1, 1.0/EQUADPriors[EQIndex]/EQUADPriors[EQIndex]
					Prior = 1.0/EQUADPriors[EQIndex]/EQUADPriors[EQIndex]
					hess_dense[index,index]  += Prior#/np.sum(self.EQUADInfo==self.EQUADInfo[i])

			if(self.fitECORRSignal == True):
				EpochIndex = self.EpochIndex[i]
				ECORRIndex = np.int(self.ECORRInfo[EpochIndex])
				index = self.ParamDict['ECORRSignal'][0] + self.EpochIndex[i] - self.DiagParams
				#print index

				if(self.ECORRModel[ECORRIndex] == -1):
					hess_dense[index,index]  += 1.0/len(self.ChansPerEpoch[EpochIndex])

				if(self.ECORRModel[ECORRIndex] == 0):
					hess_dense[index,index]  += 1.0/len(self.ChansPerEpoch[EpochIndex])

				if(self.ECORRModel[ECORRIndex] == 1):
					Prior = 1.0/ECORRPriors[ECORRIndex]/ECORRPriors[ECORRIndex]
					hess_dense[index,index]  += Prior/len(self.ChansPerEpoch[EpochIndex]) #/np.sum(self.EQUADInfo==self.EQUADInfo[i])


			pcount=LinearSize

			######################Now add all non-linear parameters to Hessian##############################


			if(self.fitEQUADPrior == True):
				index = self.ParamDict['EQUADPrior'][0] - self.DiagParams
				EQIndex = np.int(self.EQUADInfo[i])

				if(self.EQUADModel[EQIndex] == -1):
					hess_dense[index,index]  += 15.0/np.sum(self.EQUADInfo==EQIndex)

			if(self.fitECORRPrior == True):
	
				EpochIndex = self.EpochIndex[i]
				ECORRIndex = np.int(self.ECORRInfo[EpochIndex])
				index = self.ParamDict['ECORRPrior'][0] + ECORRIndex - self.DiagParams
				#print i, EpochIndex, ECORRIndex, index, np.shape(hess_dense),15.0/np.sum(self.ECORRInfo==ECORRIndex)/len(self.ChansPerEpoch[EpochIndex])
	
				if(self.ECORRModel[ECORRIndex] == -1):
					hess_dense[index,index]  += 15.0/np.sum(self.ECORRInfo==ECORRIndex)/len(self.ChansPerEpoch[EpochIndex])
			
				if(self.ECORRModel[ECORRIndex] == 0):
					hess_dense[index,index]  += 15.0/np.sum(self.ECORRInfo==ECORRIndex)/len(self.ChansPerEpoch[EpochIndex])
			
				if(self.ECORRModel[ECORRIndex] == 1):
					hess_dense[index,index]  += 100.0/np.sum(self.ECORRInfo==ECORRIndex)/len(self.ChansPerEpoch[EpochIndex])

			if(self.fitScatter == True):

				index = self.ParamDict['Scattering'][0] - self.DiagParams
				for c in range(self.NScatterEpochs):
					if(c in self.ScatterInfo[i]):


						tau = ScatteringParameters[c]
						f = np.linspace(1,self.NFBasis,self.NFBasis)/self.FoldingPeriods[i]
						w = 2.0*np.pi*f
						ISS = 1.0/(self.psr.ssbfreqs()[i]**ScatterFreqScale/self.ScatterRefFreq**(ScatterFreqScale))
						ISS2 = 1.0/(self.psr.ssbfreqs()[i]**ScatterFreqScale/10.0**(9.0*ScatterFreqScale))
						#ISS = 1.0/((self.psr.ssbfreqs()[i]**ScatterFreqScale)/(self.ScatterRefFreq**(ScatterFreqScale)))
						#print i, self.psr.freqs[i], ISS, 1.0/ISS, ISS2, tau*ISS
						RConv, IConv = self.ConvolveExp(f, tau*ISS)

						RProf = NoScatterS[:self.NFBasis]*MLAmp
						IProf = NoScatterS[self.NFBasis:]*MLAmp

						RConfProf = RConv*RProf - IConv*IProf
						IConfProf = IConv*RProf + RConv*IProf

						pnoise = np.sqrt(Noise)[:self.NFBasis]

						HessDenom = 1.0/(1.0 + tau**2*w**2*ISS**2)**3
						GradDenom = 1.0/(1.0 + tau**2*w**2*ISS**2)**2

						Reaself = (RollData[:self.NFBasis] - RProf*RConv + IProf*IConv)

						'''
						#plt.plot(np.linspace(0,1, self.NFBasis), RollData[:self.NFBasis]-(RProf*RConv - IProf*IConv))
						plt.plot(np.linspace(0,1, self.NFBasis), RProf*RConv - IProf*IConv)
						plt.plot(np.linspace(0,1, self.NFBasis), RProf)
						plt.show()
						'''
						RealGrad = 2*tau**2*ISS**2*w**2*np.log(10.0)*GradDenom*RProf + tau*ISS*w*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)*GradDenom*IProf
						RealHess = -(4*tau**2*ISS**2*w**2*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)**2)*HessDenom*RProf - tau*ISS*w*(1+tau**2*ISS**2*w**2*(tau**2*ISS**2*w**2 - 6))*np.log(10.0)**2*HessDenom*IProf

						FullRealHess = 1*(RealHess*Reaself + RealGrad**2)*(1.0/pnoise**2)

						ImagFunc = (RollData[self.NFBasis:] - RProf*IConv - IProf*RConv)
						ImagGrad = 2*tau**2*ISS**2*w**2*np.log(10.0)*GradDenom*IProf - tau*ISS*w*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)*GradDenom*RProf
						ImagHess = -(4*tau**2*ISS**2*w**2*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)**2)*HessDenom*IProf + tau*ISS*w*(1+tau**2*ISS**2*w**2*(tau**2*ISS**2*w**2 - 6))*np.log(10.0)**2*HessDenom*RProf



						FullImagHess = 1*(ImagHess*ImagFunc + ImagGrad**2)*(1.0/pnoise**2)


						profhess = np.zeros(2*self.NFBasis)
						profhess[:self.NFBasis] = FullRealHess
						profhess[self.NFBasis:] = FullImagHess

						profgrad = np.zeros(2*self.NFBasis)
						profgrad[:self.NFBasis] = RealGrad*(1.0/pnoise)
						profgrad[self.NFBasis:] = ImagGrad*(1.0/pnoise)

						LinearScatterCross = np.dot(HessMatrix, profgrad)



						hess_dense[index+c,index+c] += np.sum(profhess)
			


			
						if(self.fitScatterStepSize != 0):
							hess_dense[index+c,index+c] = 1.0/self.fitScatterStepSize**2
							cov_diag[self.DiagParams+index+c] = 1.0/self.fitScatterStepSize**2
		
			
						SLinCount = 0
						for k1 in range(len(self.ParamDict.keys())):
							key1 = self.ParamDict.keys()[k1]
							if(self.ParamDict[key1][5] == 1):

								Np1 = self.ParamDict[key1][1] - self.ParamDict[key1][0]
								index1 = self.ParamDict[key1][0] - self.DiagParams

								if(key1 == 'PAmps' or key1 == 'EQUADSignal'): 
									index1 += i
									Np1 = 1

								if(key1 == 'ECORRSignal'):
									index1 += self.EpochIndex[i]
									Np1 = 1
					
					

								#print "param1: ", key1, Np1, index1
								hess_dense[index1:index1+Np1, index+c] += -self.ScatterCrossTerms[c]*LinearScatterCross[SLinCount:SLinCount+Np1]
								hess_dense[index+c, index1:index1+Np1] += -self.ScatterCrossTerms[c]*LinearScatterCross[SLinCount:SLinCount+Np1]
			
								SLinCount += Np1
		
			
						cov_diag[self.DiagParams+index+c] += np.sum(profhess)

						if(cov_diag[self.DiagParams+index+c] < 0):
							cov_diag[self.DiagParams+index+c] = 2.0
							hess_dense[index+c,index+c] = 2.0
			
			if(self.fitScatterFreqScale == True):
	
				index = self.ParamDict['ScatterFreqScale'][0] - self.DiagParams
				hess_dense[index,index] = 5000.0
				cov_diag[self.DiagParams+index] = 5000.0

		print "likelihood", like
		#return hess_dense
		if(diagonalGHS == False):		
			#Now do EVD on the dense part of the matrix


                        if(self.BaselineNoiseParams > 0):
                                for i in range(self.NToAs):
                                        V2, M2 = sl.eigh(self.BLNHess[i])
                                        cov_diag[self.BLNList[i]] = V2
                                        self.BLNEigM[i] = M2
                                        if(np.min(V2) < 1):
                                                print "Poorly formed BLN Hessian", i, " using diagonal approximation\n"
						self.BLNEigM[i] = np.eye(self.BaselineNoiseParams)
						DHess = copy.copy(self.BLNHess[i].diagonal())
						#DHess.setflags(write=1)
						DHess[DHess < 1] = 1
						cov_diag[self.BLNList[i]] = DHess


			if(self.DiagParams < self.n_params):
				V, M = sl.eigh(hess_dense)

				cov_diag[self.DiagParams:] = V
	
				if(np.min(V) < 0):
					print "Negative Eigenvalues: Poorly formed Hessian\n"

			else:
				M = np.ones(1)
				hess_dense = np.ones(1)

		else:

			hess_dense = np.eye(np.shape(hess_dense)[0])
			M = copy.copy(hess_dense)

			if(self.BaselineNoiseParams > 0):
                                for i in range(self.NToAs):
                                        self.BLNEigM[i] = np.eye(self.BaselineNoiseParams)
	

		#Complete the start point by filling in extra parameters

		for k in self.ParamDict.keys():
			if(len(self.MLParameters[self.ParamDict[k][2]]) > 1):
				self.MLParameters[self.ParamDict[k][2]] = np.float64(self.MLParameters[self.ParamDict[k][2]])
			else:
				self.MLParameters[self.ParamDict[k][2]] = np.array([np.float64(self.MLParameters[self.ParamDict[k][2]])])
		
		if(self.fitBaselineNoiseAmpPrior == True):
			index=self.ParamDict['BaselineNoiseAmpPrior'][0]
			x0[index:index + self.NToAs] = self.MLParameters[self.ParamDict['BaselineNoiseAmpPrior'][2]]
		
		if(self.fitBaselineNoiseSpecPrior == True):
			index=self.ParamDict['BaselineNoiseSpecPrior'][0]
			x0[index:index + self.NToAs] = self.MLParameters[self.ParamDict['BaselineNoiseSpecPrior'][2]]

		if(self.fitPhase == True):
			index=self.ParamDict['Phase'][0]
			x0[index] = self.MLParameters[self.ParamDict['Phase'][2]][0]		

		if(self.fitPhase == True):
			index=self.ParamDict['Phase'][0]
			x0[index] = self.MLParameters[self.ParamDict['Phase'][2]][0]

		if(self.fitLinearTM == True):
			index=self.ParamDict['LinearTM'][0]
			x0[index:index + self.numTime] = self.MLParameters[self.ParamDict['LinearTM'][2]]

		if(self.fitProfile == True):
			index=self.ParamDict['Profile'][0]
			x0[index:index + NCoeff*(self.EvoNPoly+1)] = self.MLParameters[self.ParamDict['Profile'][2]]		

		if(self.fitEQUADSignal == True):
			index=self.ParamDict['EQUADSignal'][0]
			x0[index:index + self.NToAs] = self.MLParameters[self.ParamDict['EQUADSignal'][2]]
	
		if(self.fitEQUADPrior == True):
			index=self.ParamDict['EQUADPrior'][0]
			x0[index:index + self.NumEQPriors] = self.MLParameters[self.ParamDict['EQUADPrior'][2]]
	
		if(self.fitECORRSignal == True):
			index=self.ParamDict['ECORRSignal'][0]
			x0[index:index + self.NumEpochs] = self.MLParameters[self.ParamDict['ECORRSignal'][2]]
	
		if(self.fitECORRPrior == True):
			index=self.ParamDict['ECORRPrior'][0]
			x0[index:index + self.NumECORRPriors] = self.MLParameters[self.ParamDict['ECORRPrior'][2]]

		if(self.fitScatter == True):
			index=self.ParamDict['Scattering'][0]
			x0[index:index + self.NScatterEpochs] = self.MLParameters[self.ParamDict['Scattering'][2]]
	
		if(self.fitScatterFreqScale == True):
			index=self.ParamDict['ScatterFreqScale'][0]
			x0[index] = self.MLParameters[self.ParamDict['ScatterFreqScale'][2]]


		
		return x0, cov_diag, M, hess_dense





	def ConvolveExp(self, f, tau, returngrad=False, returnComp = False):

		w = 2.0*np.pi*f
		Real = 1.0/(w**2*tau**2+1) 
		Imag = -1*w*tau/(w**2*tau**2+1)
	
		if(returnComp == True):
			return Real - 1j*Imag

		if(returngrad==False):
			return Real, Imag
		else:
			grad = (1.0 - tau**2*w**2)/(tau**2*w**2 + 1)**2 - 1j*tau*w/(tau**2*w**2 + 1)**2 

	def GetScatteringParams(self, mode='parfile', flag = None, timestep = None):

		SXParamList = [None]*self.NToAs

		if(mode == 'parfile'):
			allparams=self.psr.pars(which='set')
			SXList = [sx for sx in allparams if 'SX_' in sx]

			self.NScatterEpochs = len(SXList)

		
			for i in range(self.NScatterEpochs):
				mintime = self.psr['SXR1_'+SXList[i][-4:]].val
				maxtime = self.psr['SXR2_'+SXList[i][-4:]].val

				select_indices = np.where(np.logical_and( self.psr.stoas < maxtime, self.psr.stoas >=mintime))[0]

				for index in select_indices:
					#print index, self.psr.stoas[index], SXList[i]
					if(SXParamList[index] == None):
						SXParamList[index] = [i]
					else:
						SXParamList[index].append(i)

		if(mode == 'flag'):
			scatterflags = np.unique(self.psr.flagvals(flag))
			self.NScatterEpochs = len(scatterflags)
			for i in range(self.NScatterEpochs):
				select_indices = np.where(self.psr.flagvals(flag) ==  scatterflags[i])[0] 
				for index in select_indices:
					print index, self.psr.stoas[index], scatterflags[i]
					if(SXParamList[index] == None):
						SXParamList[index] = [i]
					else:
						SXParamList[index].append(i)

		if(mode == 'time'):
			startMJD = np.min(self.psr.stoas)
			endMJD = np.max(self.psr.stoas)
			self.NScatterEpochs = (endMJD-startMJD)/time
			for i in range(self.NScatterEpochs):
				select_indices = np.where(np.logical_and( self.psr.stoas < startMJD+(i+1)*time, self.psr.stoas >= startMJD+i*time))[0]
				for index in select_indices:
					print index, self.psr.stoas[index]
					if(SXParamList[index] == None):
						SXParamList[index] = [i]
					else:
						SXParamList[index].append(i)

		
		return np.array(SXParamList)


#def PrintEpochParams(time):
		

	def PrintEpochParams(self, time=30, string ='DMX', fit = 1):

		totalinEpochs = 0
		stoas = self.psr.stoas
		mintime = stoas.min()
		maxtime = stoas.max()

		NEpochs = np.ceil((maxtime-mintime)/time)
		Epochs=mintime - time*0.01 + np.arange(NEpochs+1)*time #np.linspace(mintime-time*0.01, maxtime+time*0.01, int(NEpochs+3))
		EpochList = []
		for i in range(len(Epochs)-1):
			select_indices = np.where(np.logical_and( stoas < Epochs[i+1], stoas >= Epochs[i]))[0]
			if(len(select_indices) > 0):
				print "There are ", len(select_indices), " profiles in Epoch ", i
				totalinEpochs += len(select_indices)
				EpochList.append(i)
		print "Total profiles in Epochs: ", totalinEpochs, " out of ", self.psr.nobs
		EpochList = np.unique(np.array(EpochList))
		for i in range(len(EpochList)):
			if(i < 9):
				print string+"_000"+str(i+1)+" 0 "+str(fit)
				print string+"R1_000"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_000"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_000"+str(i+1)+" 0.20435656\n"

			if(i < 99 and i >= 9):
				print string+"_00"+str(i+1)+" 0 "+str(fit)
				print string+"R1_00"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_00"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_00"+str(i+1)+" 0.20435656\n"

			if(i < 999 and i >= 99):
				print string+"_0"+str(i+1)+" 0 "+str(fit)
				print string+"R1_0"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_0"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_0"+str(i+1)+" 0.20435656\n"

			if(i < 9999 and i >= 999):
				print string+"_"+str(i+1)+" 0 "+str(fit)
				print string+"R1_"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_"+str(i+1)+" 0.20435656\n"


	def WaterFallPlot(self, ML):

		self.doplot = True
		Res, Data, Model =self.FFTMarginLogLike(ML)
		self.doplot = False
		Res=np.array(Res).T
		Data=np.array(Data).T	
		Model=np.array(Model).T

		x=np.float64(self.psr.stoas)
		tdiff = np.max(self.psr.stoas) - np.min(self.psr.stoas)
		aspectgoal = 16.0/6
		aspect=tdiff/aspectgoal

		fig, (ax1, ax2) = plt.subplots(2,1)

		im1 = ax1.imshow(np.log10(np.abs(Data)), interpolation="none", extent=(np.min(x), np.max(x), 0, 1), aspect=aspect)
		#divider1 = make_axes_locatable(ax1)
		# Append axes to the right of ax3, with 20% width of ax3
		#cax1 = divider1.append_axes("right", size="20%", pad=0.05)
		# Create colorbar in the appended axes
		# Tick locations can be set with the kwarg `ticks`
		# and the format of the ticklabels with kwarg `format`
		#cbar1 = plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.2), format="%.2f")

		im2 = ax2.imshow(np.abs(Res), interpolation="none", extent=(np.min(x), np.max(x), 0, 1), aspect=aspect)
		#divider2 = make_axes_locatable(ax2)
		# Append axes to the right of ax3, with 20% width of ax3
		#cax2 = divider2.append_axes("right", size="20%", pad=0.05)
		# Create colorbar in the appended axes
		# Tick locations can be set with the kwarg `ticks`
		# and the format of the ticklabels with kwarg `format`
		#cbar2 = plt.colorbar(im2, cax=cax2, ticks=MultipleLocator(0.2), format="%.2f")

		plt.show()


	def GHSGPULike(self, ndim, x, like, g):

		params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))
		ndims = ndim[0]

		#Send relevant parameters to physical coordinates for likelihood

		if(self.BaselineNoiseParams > 0):
			for i in range(self.NToAs):
				DenseParams = params[self.BLNList[i]]
				PhysParams = np.dot(self.BLNEigM[i], DenseParams)
				params[self.BLNList[i]] = PhysParams

		if(self.DiagParams < ndims):
			DenseParams = params[self.DiagParams:]
			PhysParams = np.dot(self.EigM, DenseParams)
			params[self.DiagParams:] = PhysParams

		likeval, grad = self.GPULike(ndims, params)


                if(self.BaselineNoiseParams > 0):
                        for i in range(self.NToAs):
                                DenseGrad = copy.copy(grad[self.BLNList[i]])
                                PrincipleGrad = np.dot(self.BLNEigM[i].T, DenseGrad)
                                grad[self.BLNList[i]] = PrincipleGrad


		if(self.DiagParams < ndims):
			DenseGrad = copy.copy(grad[self.DiagParams:])
			PrincipleGrad = np.dot(self.EigM.T, DenseGrad)
			grad[self.DiagParams:] = PrincipleGrad
		    
		#print("like:", like[0], "grad", PrincipleGrad, DenseGrad)
		for i in range(ndim[0]):
			g[i] = grad[i]

		like[0] = likeval


	def GPULike(self, ndim, params):

		####################Cublas Parameters########################################


		alpha = np.float64(1.0)
		beta = np.float64(0.0)


		####################Copy Parameter Vector#################################

		#params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))


		#Send relevant parameters to physical coordinates for likelihood

		#DenseParams = params[self.DiagParams:]
		#PhysParams = np.dot(self.EigM, DenseParams)
		#params[self.DiagParams:] = PhysParams
		#print("Phys Params: ", PhysParams)

		#grad = np.zeros(ndim[0])
		grad = np.zeros(ndim)

		#like[0] = 0
		like = 0

		####################Get Parameters########################################
	
		if(self.fitPAmps == True):
			index=self.ParamDict['PAmps'][0]
			ProfileAmps = params[index:index+self.NToAs]
		else:
			ProfileAmps = self.MLParameters[self.ParamDict['PAmps'][2]]

		if(self.fitPNoise == True):
			index=self.ParamDict['PNoise'][0]
			ProfileNoise = params[index:index+self.NToAs]*params[index:index+self.NToAs]
		else:
			ProfileNoise = np.float64(self.MLParameters[self.ParamDict['PNoise'][2]]*self.MLParameters[self.ParamDict['PNoise'][2]])

		gpu_Amps = gpuarray.to_gpu(np.float64(ProfileAmps))
		gpu_Noise = gpuarray.to_gpu(np.float64(ProfileNoise)) 

		if(self.fitPhase == True):
			index=self.ParamDict['Phase'][0]
			Phase = params[index]
			phasePrior = 0.5*(Phase-self.MeanPhase)*(Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
			phasePriorGrad = (Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
		else:
			Phase = self.MLParameters[self.ParamDict['Phase'][2]][0]
			phasePrior = 0
			phasePriorGrad = 0

		if(self.incLinearTM == True):
			if(self.fitLinearTM == True):
				index = self.ParamDict['LinearTM'][0]
				TimingParameters = params[index:index+self.numTime]
			else:
				TimingParameters = self.MLParameters[self.ParamDict['LinearTM'][2]]

		NCoeff = self.TotCoeff-1

		if(self.incProfile == True):
			if(self.fitProfile == True):
				index = self.ParamDict['Profile'][0]
				ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
				ShapeAmps[0][0] = 1
				ShapeAmps[1:]=params[index:index + NCoeff*(self.EvoNPoly+1)].reshape([NCoeff,(self.EvoNPoly+1)])
				if(self.EvoNPoly == 0):
					ShapeAmps = ShapeAmps.T.flatten()
				else:
					ShapeAmps = ShapeAmps.flatten()
			else:
				ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
				ShapeAmps[0][0] = 1
				ShapeAmps[1:] = self.MLParameters[self.ParamDict['Profile'][2]].reshape([NCoeff,(self.EvoNPoly+1)])

				if(self.EvoNPoly == 0):
					ShapeAmps = ShapeAmps.T.flatten()
				else:
					ShapeAmps = ShapeAmps.flatten()

			ShapeAmps_GPU = gpuarray.to_gpu(ShapeAmps)


		if(self.incScatter == True):
			if(self.fitScatter == True):
				index=self.ParamDict['Scattering'][0]
				ScatteringParameters = params[index:index+self.NScatterEpochs]
				if(self.fitScatterPrior == 0):
					for i in range(self.NScatterEpochs):
						if(ScatteringParameters[i] < -6):
							like += -np.log(10.0)*(ScatteringParameters[i]+6)
							grad[i+index] += -np.log(10.0)
				if(self.fitScatterPrior == 1):
                                        for i in range(self.NScatterEpochs):
						like += -np.log(10.0)*(ScatteringParameters[i])
						grad[i+index] += -np.log(10.0)
				
				ScatteringParameters = 10.0**ScatteringParameters
				ScatteringParameters_GPU = gpuarray.to_gpu(ScatteringParameters)
			else:
				ScatteringParameters = 10.0**self.MLParameters[self.ParamDict['Scattering'][2]]
				ScatteringParameters_GPU = gpuarray.to_gpu(ScatteringParameters)


			if(self.fitScatterFreqScale == True):
				index=self.ParamDict['ScatterFreqScale'][0]
				ScatterFreqScale = params[index]
			else:
				ScatterFreqScale = self.MLParameters[self.ParamDict['ScatterFreqScale'][2]]
		
		if(self.incEQUAD == True):
		
			if(self.fitEQUADPrior == True):
				index=self.ParamDict['EQUADPrior'][0]
				EQUADPriors  = np.zeros(self.NumEQPriors)
				EQUADPriors[:] =  params[index:index+self.NumEQPriors]
				for i in range(self.NumEQPriors):
					if(EQUADPriors[i] < -10):
						like[0] += -np.log(10.0)*(EQUADPriors[i]+10)
						grad[i+index] += -np.log(10.0)
			else:
				EQUADPriors  = copy.copy(self.MLParameters[self.ParamDict['EQUADPrior'][2]])

			EQUADPriors = 10.0**EQUADPriors
		
			if(self.fitEQUADSignal == True):
				index=self.ParamDict['EQUADSignal'][0]
				EQUADSignal = params[index:index+self.NToAs]
	
				for i in range(self.NumEQPriors):
					EQIndicies = np.where(self.EQUADInfo==i)[0]
					Prior = EQUADPriors[i]
					if(self.EQUADModel[i] == -1 or self.EQUADModel[i] == 0):
						like += 0.5*np.sum(EQUADSignal[EQIndicies]*EQUADSignal[EQIndicies])
						grad[EQIndicies+index] += EQUADSignal[EQIndicies]
						EQUADSignal[EQIndicies] *= Prior

					if(self.EQUADModel[i] == 1):
						like += 0.5*np.sum(EQUADSignal[EQIndicies]*EQUADSignal[EQIndicies])/Prior/Prior + 0.5*len(EQIndicies)*np.log(Prior*Prior)
						grad[EQIndicies+index] += EQUADSignal[EQIndicies]/Prior/Prior

			else:
				EQUADSignal = copy.copy(self.MLParameters[self.ParamDict['EQUADSignal'][2]])
				for i in range(self.NumEQPriors):
					EQIndicies = np.where(self.EQUADInfo==i)[0]
					Prior = EQUADPriors[i]
					if(self.EQUADModel[i] == -1 or self.EQUADModel[i] == 0):
						like += 0.5*np.sum(EQUADSignal[EQIndicies]*EQUADSignal[EQIndicies])
						EQUADSignal[EQIndicies] *= Prior
			
					if(self.EQUADModel[i] == 1):
						like += 0.5*np.sum(EQUADSignal[EQIndicies]*EQUADSignal[EQIndicies])/Prior/Prior + 0.5*len(EQIndicies)*np.log(Prior*Prior)
			
			self.TimeJitterSignal_GPU = gpuarray.to_gpu(EQUADSignal)
	

		if(self.incECORR == True):
		
			if(self.fitECORRPrior == True):
				index=self.ParamDict['ECORRPrior'][0]
				ECORRPriors  = params[index:index+self.NumECORRPriors]
				for i in range(self.NumECORRPriors):
					if(ECORRPriors[i] < -10):
						like += -np.log(10.0)*(ECORRPriors[i]+10)
						grad[i+index] += -np.log(10.0)
			else:
				ECORRPriors  = copy.copy(self.MLParameters[self.ParamDict['ECORRPrior'][2]])

			ECORRPriors = 10.0**ECORRPriors
		
			if(self.fitECORRSignal == True):
				index=self.ParamDict['ECORRSignal'][0]
				ECORRSignal = params[index:index+self.NumEpochs]
	
				for i in range(self.NumECORRPriors):
					ECORRIndicies = np.where(self.ECORRInfo==i)[0]
					Prior = ECORRPriors[i]
					if(self.ECORRModel[i] == -1 or self.ECORRModel[i] == 0):
						like += 0.5*np.sum(ECORRSignal[ECORRIndicies]*ECORRSignal[ECORRIndicies])
						grad[ECORRIndicies+index] += ECORRSignal[ECORRIndicies]
						ECORRSignal[ECORRIndicies] *= Prior

					if(self.ECORRModel[i] == 1):
						like += 0.5*np.sum(ECORRSignal[ECORRIndicies]*ECORRSignal[ECORRIndicies])/Prior/Prior + 0.5*len(ECORRIndicies)*np.log(Prior*Prior)
						grad[ECORRIndicies+index] += ECORRSignal[ECORRIndicies]/Prior/Prior

			else:
				ECORRSignal = copy.copy(self.MLParameters[self.ParamDict['ECORRSignal'][2]])
				for i in range(self.NumECORRPriors):
					ECORRIndicies = np.where(self.ECORRInfo==i)[0]
					Prior = ECORRPriors[i]
					if(self.ECORRModel[i] == -1 or self.ECORRModel[i] == 0):
						like += 0.5*np.sum(ECORRSignal[ECORRIndicies]*ECORRSignal[ECORRIndicies])
						ECORRSignal[ECORRIndicies] *= Prior
			
					if(self.ECORRModel[i] == 1):
						like += 0.5*np.sum(ECORRSignal[ECORRIndicies]*ECORRSignal[ECORRIndicies])/Prior/Prior + 0.5*len(ECORRIndicies)*np.log(Prior*Prior)
			
			JitterSignal = ECORRSignal[self.EpochIndex]

			self.TimeJitterSignal_GPU = gpuarray.to_gpu(JitterSignal)	


		if(self.incBaselineNoise == True):

			if(self.fitBaselineNoiseAmpPrior == True):
				index=self.ParamDict['BaselineNoiseAmpPrior'][0]
				BaselineNoisePriorAmps  = params[index:index+self.NToAs]
				for i in range(self.NToAs):
					if(BaselineNoisePriorAmps[i] < -10 or self.BaselineNoisePrior[i] == 1 or self.FindML == True):
						like += -np.log(10.0)*(BaselineNoisePriorAmps[i]+10)
						grad[i+index] += -np.log(10.0)
			else:
				BaselineNoisePriorAmps  = copy.copy(self.MLParameters[self.ParamDict['BaselineNoiseAmpPrior'][2]])


			if(self.fitBaselineNoiseSpecPrior == True):
				index=self.ParamDict['BaselineNoiseSpecPrior'][0]
				BaselineNoisePriorSpecs  = params[index:index+self.NToAs]
				for i in range(self.NToAs):
                                        if(BaselineNoisePriorSpecs[i] < -7):
                                                like += -(BaselineNoisePriorSpecs[i]+7)
                                                grad[i+index] += -1
					if(BaselineNoisePriorSpecs[i] > 10):
                                                like += (BaselineNoisePriorSpecs[i]-10)
                                                grad[i+index] += 1
			else:
				BaselineNoisePriorSpecs  = copy.copy(self.MLParameters[self.ParamDict['BaselineNoiseSpecPrior'][2]])


	
			self.BaselineNoiseAmps_GPU = gpuarray.to_gpu(BaselineNoisePriorAmps)
			self.BaselineNoiseSpecs_GPU = gpuarray.to_gpu(BaselineNoisePriorSpecs)

			


		if(self.incLinearTM == True):
			TimingParameters_GPU = gpuarray.to_gpu(np.float64(TimingParameters))
			cublas.cublasDgemv(self.CUhandle, 't',  self.numTime, self.NToAs, alpha, self.DesignMatrix_GPU.gpudata, self.numTime, TimingParameters_GPU.gpudata, 1, beta, self.TimeSignal_GPU.gpudata, 1)


		####################Calculate Profile Amplitudes########################################

		block_size = 128
		grid_size = int(np.ceil(self.TotCoeff*self.NToAs*1.0/block_size))
		self.GPUPrepLikelihood(self.ProfAmps_GPU, ShapeAmps_GPU, self.gpu_SSBFreqs, np.int32(self.NToAs), np.int32(self.TotCoeff), np.int32(self.EvoNPoly), np.float64(self.EvoRefFreq), grid=(grid_size,1), block=(block_size,1,1))



		####################Calculate Phase Offsets########################################


		block_size = 128
		grid_size = int(np.ceil(self.NToAs*1.0/block_size))
		self.GPUBinTimes(self.gpu_ShiftedBinTimes, self.gpu_NBins, np.float64(Phase*self.ReferencePeriod), self.TimeSignal_GPU, self.TimeJitterSignal_GPU, self.gpu_FoldingPeriods, np.float64(self.InterpolatedTime), self.gpu_xS, self.gpu_InterpBins, self.gpu_WBTs, self.gpu_RollBins, self.InterpPointers_GPU, self.i_arr_gpu, self.InterpJPointers_GPU, self.JPointers_gpu, np.int32(self.NToAs), np.int32(np.shape(self.InterpFBasis)[0]), grid=(grid_size,1), block=(block_size,1,1))


		####################GPU Batch submit DGEMM for profile and jitter########################################

		#if we have M = (n x l x m) and N = (n x m x k), MN = O = (n x l x k) the interface to gemmbatched is:
		#cublas.cublasDgemmBatched(h, 'n','n', k, l, m, alpha, MatrixN, k, MatrixM, m, beta, MatrixO, k, n)


		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 2*self.NFBasis, self.TotCoeff, alpha, self.ProfAmps_Pointer.gpudata, 1, self.i_arr_gpu.gpudata, self.TotCoeff, beta, self.Signal_Pointer.gpudata, 1, self.NToAs)

		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 2*self.NFBasis, self.TotCoeff, alpha, self.ProfAmps_Pointer.gpudata, 1, self.JPointers_gpu.gpudata, self.TotCoeff, beta, self.JSignal_Pointer.gpudata, 1, self.NToAs)

		####################Rotate Data########################################

		block_size = 128
		grid_size = int(np.ceil(self.NToAs*self.NFBasis*1.0/block_size))
		Step = np.int32(self.NToAs*self.NFBasis)

		self.GPURotateData(self.gpu_Data,  self.gpu_Freqs, self.gpu_RollBins, self.gpu_ToAIndex, self.gpu_RolledData, Step, grid=(grid_size,1), block=(block_size,1,1))


		################Scatter if in the model##################################

		#NoScatterM = self.Signal_GPU.get()
		#RData = self.gpu_RolledData.get()



		if(self.incScatter == True):

			self.GPUScatter(self.Signal_GPU, self.JSignal_GPU, self.ScatterGrads_GPU, ScatteringParameters_GPU, self.gpu_Freqs, self.gpu_SSBFreqs, Step, self.gpu_ScatterIndex, self.gpu_ToAIndex, self.gpu_SignalIndex, gpu_Amps, self.gpu_NBins, self.gpu_FoldingPeriods, np.int32(self.NFBasis),np.int32(self.TotCoeff), self.ScatteredBasis_GPU, self.InterpFBasis_GPU, self.gpu_InterpBins, ScatterFreqScale, np.float64(self.ScatterRefFreq), grid=(grid_size,1), block=(block_size,1,1))
	
		'''		
		ScatterM = self.Signal_GPU.get()
		ScatterJ = self.JSignal_GPU.get()
		for i in range(self.NToAs):

			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RData[i*self.NFBasis:(i+1)*self.NFBasis] 
			RollData[self.NFBasis:] = RData[(self.NToAs + i)*self.NFBasis:(self.NToAs + i+1)*self.NFBasis]

			PSignal = NoScatterM[i,:,0]*ProfileAmps[i]
			PSignal2 = ScatterM[i,:,0]*ProfileAmps[i]
			plt.plot(np.linspace(0,2,2*self.NFBasis), RollData, color='black')
			plt.plot(np.linspace(0,2,2*self.NFBasis), PSignal, color='red')
			plt.plot(np.linspace(0,2,2*self.NFBasis), PSignal2, color='blue')
			plt.xlabel('Frequency')
			plt.ylabel('Profile Amplitude')
			plt.show()

			plt.plot(np.linspace(0,2,2*self.NFBasis), RollData-PSignal2, color='black')
			plt.xlabel('Frequency')
			plt.ylabel('Profile Amplitude')
			plt.show()

			bd = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
			bd[1:self.NFBasis+1] = RollData[:self.NFBasis] + 1j*RollData[self.NFBasis:]
			bdt = np.fft.irfft(bd)

			bm = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
			bm[1:self.NFBasis+1] = PSignal2[:self.NFBasis] + 1j*PSignal2[self.NFBasis:]
			bmt = np.fft.irfft(bm)

			plt.plot(np.linspace(0,1,self.Nbins[i]), bdt, color='black')
			plt.plot(np.linspace(0,1,self.Nbins[i]), bmt, color='red')
			plt.xlabel('Phase')
			plt.ylabel('Profile Amplitude')
			plt.show()
			plt.plot(np.linspace(0,1,self.Nbins[i]),bdt-bmt)
			plt.xlabel('Phase')
			plt.ylabel('Profile Residuals')
			plt.show()
		'''


		####################Compute Chisq########################################

		if(self.ReturnProfile == True):
			return self.Signal_GPU.get(), self.gpu_RolledData.get()


		TotBins = np.int32(2*self.NToAs*self.NFBasis)
		grid_size = int(np.ceil(self.NToAs*self.NFBasis*2.0/block_size))	


		if(self.incBaselineNoise == False):

			self.GPUGetRes(self.gpu_ResVec, self.gpu_NResVec, self.gpu_RolledData, self.Signal_GPU, gpu_Amps, gpu_Noise, self.gpu_ToAIndex, 	self.gpu_SignalIndex, TotBins, grid=(grid_size,1), block=(block_size,1,1))


			cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.ResVec_pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.Chisqs_Pointer.gpudata, 1, self.NToAs)

			ChisqVec = self.Chisqs_GPU.get()[:,0,0]
			gpu_Chisq=np.sum(ChisqVec)
			like += 0.5*gpu_Chisq + 0.5*2*self.NFBasis*np.sum(np.log(ProfileNoise))

		else:


			self.GPUGetBaselineRes(self.gpu_ResVec, self.gpu_NResVec, self.gpu_RolledData, self.Signal_GPU, gpu_Amps, gpu_Noise, self.gpu_ToAIndex, self.gpu_SignalIndex, TotBins, np.int32(self.NFBasis), self.BaselineNoiseAmps_GPU, self.BaselineNoiseSpecs_GPU, self.BaselineNoiseLike_GPU, np.int32(self.BaselineNoiseRefFreq), grid=(grid_size,1), block=(block_size,1,1))

			cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.ResVec_pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.Chisqs_Pointer.gpudata, 1, self.NToAs)

			ChisqVec = self.Chisqs_GPU.get()[:,0,0]
			gpu_Chisq=np.sum(ChisqVec)

                        block_size = 128
                        grid_size = int(np.ceil(self.NToAs*1.0/block_size))
                        self.GPUGetBaselineGrads(self.gpu_ResVec, self.gpu_NResVec, self.gpu_RolledData, self.Signal_GPU, gpu_Amps, gpu_Noise, self.gpu_ToAIndex, self.gpu_SignalIndex, TotBins, np.int32(self.NFBasis), self.BaselineNoiseAmps_GPU, self.BaselineNoiseSpecs_GPU, self.BaselineNoiseLike_GPU, np.int32(self.BaselineNoiseRefFreq), np.int32(self.NToAs), grid=(grid_size,1), block=(block_size,1,1))

			NDet  = np.sum(self.BaselineNoiseLike_GPU.get())
			like += 0.5*gpu_Chisq + 0.5*NDet



		####################Calculate Gradients for amplitude and noise levels########################################

		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.Signal_Pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.AmpGrads_Pointer.gpudata, 1, self.NToAs)

		if(self.fitPAmps == True):
			index=self.ParamDict['PAmps'][0]
			grad[index:index+self.NToAs] = -1*self.AmpGrads_GPU.get()[:,0,0]

		if(self.fitPNoise == True):

			index=self.ParamDict['PNoise'][0]

			if(self.incBaselineNoise == False):				
				grad[index:index+self.NToAs] = (-ChisqVec+2*self.NFBasis)/np.sqrt(ProfileNoise)
			else:
				grad[index:index+self.NToAs] = gpu_Noise.get()
				'''
				RVec = self.gpu_ResVec.get()[:,:,0]
				NVec = self.gpu_NResVec.get()[:,0,:]

				for i in range(self.NToAs):
					BLRefF = self.BaselineNoiseRefFreq
		                        BLNFreqs = np.zeros(2*self.NFBasis)
		                        BLNFreqs[:self.NFBasis] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)
		                        BLNFreqs[self.NFBasis:] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)

					Amp=10.0**(2*BaselineNoisePriorAmps[i])
					Spec = BaselineNoisePriorSpecs[i]
					BLNPower = Amp*pow(BLNFreqs, -Spec)

					BLNPower[self.NFBasis-5:self.NFBasis] = 0
					BLNPower[-5:] = 0

					NGrad = np.sum((np.sqrt(ProfileNoise[i])/(BLNPower + ProfileNoise[i]))*(1 - RVec[i]*NVec[i]))

					grad[index+i] = NGrad
				'''

		if(self.fitBaselineNoiseAmpPrior == True or self.fitBaselineNoiseSpecPrior == True):


			#block_size = 128
			#grid_size = int(np.ceil(self.NToAs*1.0/block_size))

			#self.GPUGetBaselineGrads(self.gpu_NResVec, self.BaselineNoiseSignal_GPU, self.BaselineNoiseAmps_GPU, self.BaselineNoiseSpecs_GPU, self.BaselineNoiseLike_GPU, Step, np.int32(self.NToAs), np.int32(self.NFBasis), np.int32(self.BaselineNoiseRefFreq), grid=(grid_size,1), block=(block_size,1,1))


			if(self.fitBaselineNoiseAmpPrior == True):

				index=self.ParamDict['BaselineNoiseAmpPrior'][0]
                                grad[index:index+self.NToAs] = self.BaselineNoiseAmps_GPU.get()
				
				'''
				if(self.fitPNoise == False):
					RVec = self.gpu_ResVec.get()[:,:,0]
					NVec = self.gpu_NResVec.get()[:,0,:]

				index=self.ParamDict['BaselineNoiseAmpPrior'][0]

				for i in range(self.NToAs):
					BLRefF = self.BaselineNoiseRefFreq
                                        BLNFreqs = np.zeros(2*self.NFBasis)
                                        BLNFreqs[:self.NFBasis] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)
                                        BLNFreqs[self.NFBasis:] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)

                                        Amp=10.0**(2*BaselineNoisePriorAmps[i])
                                        Spec = BaselineNoisePriorSpecs[i]
                                        BLNPower = Amp*pow(BLNFreqs, -Spec)

                                        BLNPower[self.NFBasis-5:self.NFBasis] = 0
                                        BLNPower[-5:] = 0

					Top = np.log(10.0)*BLNPower

                                        NGrad = np.sum((Top/(BLNPower + ProfileNoise[i]))*(1 - RVec[i]*NVec[i]))

					grad[index+i] = NGrad
				'''

			if(self.fitBaselineNoiseSpecPrior == True):

				index=self.ParamDict['BaselineNoiseSpecPrior'][0]
				grad[index:index+self.NToAs] = self.BaselineNoiseSpecs_GPU.get()

				'''
                                if(self.fitPNoise == False and self.fitBaselineNoiseAmpPrior == False):
                                        RVec = self.gpu_ResVec.get()[:,:,0]
                                        NVec = self.gpu_NResVec.get()[:,0,:]

				index=self.ParamDict['BaselineNoiseSpecPrior'][0]

                                for i in range(self.NToAs):
                                        BLRefF = self.BaselineNoiseRefFreq
                                        BLNFreqs = np.zeros(2*self.NFBasis)
                                        BLNFreqs[:self.NFBasis] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)
                                        BLNFreqs[self.NFBasis:] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)

                                        Amp=10.0**(2*BaselineNoisePriorAmps[i])
                                        Spec = BaselineNoisePriorSpecs[i]
                                        BLNPower = Amp*pow(BLNFreqs, -Spec)

                                        BLNPower[self.NFBasis-5:self.NFBasis] = 0
                                        BLNPower[-5:] = 0

                                        Top = -0.5*np.log(BLNFreqs)*BLNPower

                                        NGrad = np.sum((Top/(BLNPower + ProfileNoise[i]))*(1 - RVec[i]*NVec[i]))

                                        grad[index+i] = NGrad
				'''

		#block_size = 128
                #grid_size = int(np.ceil(self.NToAs*1.0/block_size))
		#self.GPUGetBaselineGrads(self.gpu_ResVec, self.gpu_NResVec, self.gpu_RolledData, self.Signal_GPU, gpu_Amps, gpu_Noise, self.gpu_ToAIndex, self.gpu_SignalIndex, TotBins, np.int32(self.NFBasis), self.BaselineNoiseAmps_GPU, self.BaselineNoiseSpecs_GPU, self.BaselineNoiseLike_GPU, np.int32(self.BaselineNoiseRefFreq), np.int32(self.NToAs), grid=(grid_size,1), block=(block_size,1,1))	

			#block_size = 128
			#grid_size = int(np.ceil(self.NToAs*1.0/block_size))

			#self.GPUGetBaselineGrads(self.gpu_NResVec, self.BaselineNoiseSignal_GPU, self.BaselineNoiseAmps_GPU, self.BaselineNoiseSpecs_GPU, self.BaselineNoiseLike_GPU, Step, np.int32(self.NToAs), np.int32(self.NFBasis), np.int32(self.BaselineNoiseRefFreq), grid=(grid_size,1), block=(block_size,1,1))


			if(self.fitBaselineNoiseAmpPrior == True):

				index=self.ParamDict['BaselineNoiseAmpPrior'][0]
                                grad[index:index+self.NToAs] = self.BaselineNoiseAmps_GPU.get()
				
				'''
				if(self.fitPNoise == False):
					RVec = self.gpu_ResVec.get()[:,:,0]
					NVec = self.gpu_NResVec.get()[:,0,:]

				index=self.ParamDict['BaselineNoiseAmpPrior'][0]

				for i in range(self.NToAs):
					BLRefF = self.BaselineNoiseRefFreq
                                        BLNFreqs = np.zeros(2*self.NFBasis)
                                        BLNFreqs[:self.NFBasis] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)
                                        BLNFreqs[self.NFBasis:] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)

                                        Amp=10.0**(2*BaselineNoisePriorAmps[i])
                                        Spec = BaselineNoisePriorSpecs[i]
                                        BLNPower = Amp*pow(BLNFreqs, -Spec)

                                        BLNPower[self.NFBasis-5:self.NFBasis] = 0
                                        BLNPower[-5:] = 0

					Top = np.log(10.0)*BLNPower

                                        NGrad = np.sum((Top/(BLNPower + ProfileNoise[i]))*(1 - RVec[i]*NVec[i]))

					grad[index+i] = NGrad
				'''

			if(self.fitBaselineNoiseSpecPrior == True):

				index=self.ParamDict['BaselineNoiseSpecPrior'][0]
				grad[index:index+self.NToAs] = self.BaselineNoiseSpecs_GPU.get()

				'''
                                if(self.fitPNoise == False and self.fitBaselineNoiseAmpPrior == False):
                                        RVec = self.gpu_ResVec.get()[:,:,0]
                                        NVec = self.gpu_NResVec.get()[:,0,:]

				index=self.ParamDict['BaselineNoiseSpecPrior'][0]

                                for i in range(self.NToAs):
                                        BLRefF = self.BaselineNoiseRefFreq
                                        BLNFreqs = np.zeros(2*self.NFBasis)
                                        BLNFreqs[:self.NFBasis] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)
                                        BLNFreqs[self.NFBasis:] = (np.linspace(1,self.NFBasis,self.NFBasis)/BLRefF)

                                        Amp=10.0**(2*BaselineNoisePriorAmps[i])
                                        Spec = BaselineNoisePriorSpecs[i]
                                        BLNPower = Amp*pow(BLNFreqs, -Spec)

                                        BLNPower[self.NFBasis-5:self.NFBasis] = 0
                                        BLNPower[-5:] = 0

                                        Top = -0.5*np.log(BLNFreqs)*BLNPower

                                        NGrad = np.sum((Top/(BLNPower + ProfileNoise[i]))*(1 - RVec[i]*NVec[i]))

                                        grad[index+i] = NGrad
				'''

		#block_size = 128
                #grid_size = int(np.ceil(self.NToAs*1.0/block_size))
		#self.GPUGetBaselineGrads(self.gpu_ResVec, self.gpu_NResVec, self.gpu_RolledData, self.Signal_GPU, gpu_Amps, gpu_Noise, self.gpu_ToAIndex, self.gpu_SignalIndex, TotBins, np.int32(self.NFBasis), self.BaselineNoiseAmps_GPU, self.BaselineNoiseSpecs_GPU, self.BaselineNoiseLike_GPU, np.int32(self.BaselineNoiseRefFreq), np.int32(self.NToAs), grid=(grid_size,1), block=(block_size,1,1))	

		####################Calculate Gradient for Phase Offset########################################

	
		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.JSignal_Pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.JitterGrads_Pointer.gpudata, 1, self.NToAs)

		JGradVec = self.JitterGrads_GPU.get()[:,0,0]*ProfileAmps

		if(self.fitPhase == True):
			index=self.ParamDict['Phase'][0]
			grad[index] = np.sum(JGradVec*self.ReferencePeriod)


		####################Calculate Gradient for Timing Model########################################

		if(self.fitLinearTM == True):
			index = self.ParamDict['LinearTM'][0]
			TimeGrad = np.dot(JGradVec, self.designMatrix)
			grad[index:index+self.numTime] = TimeGrad

	
		####################Calculate Gradient for Profile and Evolution########################################

		if(self.fitProfile == True):
	
			if(self.incScatter == False):
				cublas.cublasDgemmBatched(self.CUhandle, 'n','n', self.TotCoeff, 1, 2*self.NFBasis, alpha, self.i_arr_gpu.gpudata, self.TotCoeff, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.ShapeGrads_Pointer.gpudata, self.TotCoeff, self.NToAs)
			else:
				cublas.cublasDgemmBatched(self.CUhandle, 'n','n', self.TotCoeff, 1, 2*self.NFBasis, alpha, self.ScatteredBasis_Pointer.gpudata, self.TotCoeff, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.ShapeGrads_Pointer.gpudata, self.TotCoeff, self.NToAs)

			ShapeGradVec = self.ShapeGrads_GPU.get()[:,0,:]

			index = self.ParamDict['Profile'][0]
			for c in range(1, self.TotCoeff):
				for p in range(self.EvoNPoly+1):
					OneGrad = -1*np.sum(ShapeGradVec[:,c]*ProfileAmps*self.fvals[p])
					grad[index] = OneGrad
					index += 1

		####################Calculate Gradient for EQUAD Signal###################################################


		if(self.fitEQUADSignal == True):
			index=self.ParamDict['EQUADSignal'][0]
			for i in range(self.NumEQPriors):
				EQIndicies = np.where(self.EQUADInfo==i)[0]
				Prior = EQUADPriors[i]

				if(self.EQUADModel[i] == -1 or self.EQUADModel[i] == 0):
					grad[EQIndicies+index] += JGradVec[EQIndicies]*Prior

				if(self.EQUADModel[i] == 1):
					grad[EQIndicies+index] += JGradVec[EQIndicies]

		if(self.fitEQUADPrior == True):
			index=self.ParamDict['EQUADPrior'][0]
			for i in range(self.NumEQPriors):
				EQIndicies = np.where(self.EQUADInfo==i)[0]
				NumCorrs = len(EQIndicies)
				Prior = EQUADPriors[i]

				if(self.EQUADModel[i] == -1 or self.EQUADModel[i] == 0):
					grad[i+index] += np.log(10.0)*np.sum(JGradVec[EQIndicies]*EQUADSignal[EQIndicies])

				if(self.EQUADModel[i] == 1):
					grad[i+index] += np.log(10.0)*(EQIndicies - np.sum(EQUADSignal[EQIndicies]*EQUADSignal[EQIndicies])/Prior/Prior)

		####################Calculate Gradient for ECORR Signal###################################################


		if(self.fitECORRSignal == True or self.fitECORRPrior == True):
			ECORRGradVec = np.array([np.sum(JGradVec[self.ChansPerEpoch[i]]) for i in range(self.NumEpochs)])

			if(self.fitECORRSignal == True):
				index=self.ParamDict['ECORRSignal'][0]
				for i in range(self.NumECORRPriors):
					ECORRIndicies = np.where(self.ECORRInfo==i)[0]
					Prior = ECORRPriors[i]

					if(self.ECORRModel[i] == -1 or self.ECORRModel[i] == 0):
						grad[ECORRIndicies+index] += ECORRGradVec[ECORRIndicies]*Prior

					if(self.ECORRModel[i] == 1):
						grad[ECORRIndicies+index] += ECORRGradVec[ECORRIndicies]

			if(self.fitECORRPrior == True):
				index2=self.ParamDict['ECORRPrior'][0]
				for i in range(self.NumECORRPriors):
					ECORRIndicies = np.where(self.ECORRInfo==i)[0]
					NumCorrs = len(ECORRIndicies)
					Prior = ECORRPriors[i]
	
					if(self.ECORRModel[i] == -1 or self.ECORRModel[i] == 0):
						grad[i+index2] += np.log(10.0)*np.sum(ECORRGradVec[ECORRIndicies]*ECORRSignal[ECORRIndicies])

					if(self.ECORRModel[i] == 1):
						grad[i+index2] += np.log(10.0)*(NumCorrs - np.sum(ECORRSignal[ECORRIndicies]*ECORRSignal[ECORRIndicies])/Prior/Prior)
		
		####################Calculate Gradient for Scattering########################################			

		if(self.fitScatterFreqScale == True or self.fitScatter == True):
			
			cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.ScatterGrad_Pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.SumScatterGrad_Pointer.gpudata, 1, self.NToAs)
	
			SGrads=self.SumScatterGrad_GPU.get()[:,0,0]

			if(self.fitScatter == True):
				index=self.ParamDict['Scattering'][0]
				for c in range(self.NScatterEpochs):
					grad[index+c] = np.sum(SGrads[self.ScatterInfo[:,0]==c])

			if(self.fitScatterFreqScale == True):
				index=self.ParamDict['ScatterFreqScale'][0]
				grad[index] = np.sum(-1*SGrads*np.log(self.SSBFreqs/self.ScatterRefFreq)/np.log(10))

	
		
		#print "likelihood", like[0]
		#Add phase prior to likelihood and gradient
		if(self.fitPhase == True):
			like += phasePrior
			index=self.ParamDict['Phase'][0]
			grad[index] += phasePriorGrad

			
		#Send relevant gradients to principle coordinates for sampling

		#DenseGrad = copy.copy(grad[self.DiagParams:])
		#PrincipleGrad = np.dot(self.EigM.T, DenseGrad)
		#grad[self.DiagParams:] = PrincipleGrad
		    
		#print("like:", like[0], "grad", PrincipleGrad, DenseGrad)
		#for i in range(ndim):
		#	g[i] = grad[i]


		#print "params: ", params
		#print "grad: ", grad
		#print "like", like

		return like, grad



	def GHSCPULike(self, ndim, x, like, g):
		    
		DiagParams = 0
		if(self.fitPAmps == True):
			DiagParams += self.NToAs
		if(self.fitPNoise == True):
			DiagParams += self.NToAs
		 
		    
		params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))


		#Send relevant parameters to physical coordinates for likelihood

		DenseParams = params[DiagParams:]
		PhysParams = np.dot(self.EigM, DenseParams)
		params[DiagParams:] = PhysParams
		#print("Phys Params: ", PhysParams)
	
		#DenseParams = params[:]
		#PhysParams = np.dot(self.EigM, DenseParams)
		#params[:] = PhysParams

	
		grad=np.zeros(ndim[0])


		pcount = 0
		if(self.fitPAmps == True):
			ProfileAmps = params[pcount:pcount+self.NToAs]
			pcount += self.NToAs
		else:
			ProfileAmps = self.MLParameters[0:self.NToAs]

		if(self.fitPNoise == True):
			ProfileNoise = params[pcount:pcount+self.NToAs]*params[pcount:pcount+self.NToAs]
			pcount += self.NToAs
		else:
			ProfileNoise = self.MLParameters[self.NToAs:2*self.NToAs]*self.MLParameters[self.NToAs:2*self.NToAs]


		if(self.fitPhase == True):
			Phase = params[pcount]
			pcount += 1
		else:
			Phase = self.MeanPhase
	
		phasePrior = 0.5*(Phase-self.MeanPhase)*(Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
		phasePriorGrad = 1*(Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior


		TimingParameters = params[pcount:pcount+self.numTime]
		pcount += self.numTime



		NCoeff = self.TotCoeff-1

		ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=params[pcount:pcount + NCoeff*(self.EvoNPoly+1)].reshape([NCoeff,(self.EvoNPoly+1)])

		pcount += NCoeff*(self.EvoNPoly+1)

		if(self.fitScatter == True):
			ScatteringParameters = 10.0**params[pcount:pcount+self.NScatterEpochs]
			pcount += self.NScatterEpochs
		else:
			ScatteringParameters = 10.0**self.MLParameters[-self.NScatterEpochs:]


		TimeSignal = np.dot(self.designMatrix, TimingParameters)





		xS = self.ShiftedBinTimes - Phase*self.ReferencePeriod 

		if(self.numTime>0):
				xS -= TimeSignal
	
		xS = ( xS + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2

		InterpBins = (xS%(self.ReferencePeriod/self.Nbins[:])/self.InterpolatedTime).astype(int)
		WBTs = xS-self.InterpolatedTime*InterpBins
		RollBins=(np.round(WBTs/(self.ReferencePeriod/self.Nbins[:]))).astype(np.int)

		OneFBasis = self.InterpFBasis[InterpBins]
		OneJBasis = self.InterpJBasis[InterpBins]

		s = np.sum([np.dot(OneFBasis, ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)

		j = np.sum([np.dot(OneJBasis, ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)

		like[0] = 0
		for i in range(self.NToAs):


			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

			RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
			ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)


			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RealRoll*self.ProfileFData[i][:self.NFBasis]-ImagRoll*self.ProfileFData[i][self.NFBasis:]
			RollData[self.NFBasis:] = ImagRoll*self.ProfileFData[i][:self.NFBasis]+RealRoll*self.ProfileFData[i][self.NFBasis:]

			if(self.NScatterEpochs > 0):
				tau = np.sum(ScatteringParameters[self.ScatterInfo[i]])
				f = np.linspace(1,self.NFBasis,self.NFBasis)/self.ReferencePeriod
				w = 2.0*np.pi*f
				ISS = 1.0/(self.SSBFreqs[i]**4/10.0**(9.0*4.0))

				RConv, IConv = self.ConvolveExp(f, tau*ISS)

				RProf = s[i][:self.NFBasis]
				IProf = s[i][self.NFBasis:]

				RConfProf = RConv*RProf - IConv*IProf
				IConfProf = IConv*RProf + RConv*IProf

				PAmp = ProfileAmps[i]
				GradDenom = 1.0/(1.0 + tau**2*w**2*ISS**2)**2
				RealSGrad = 2*tau**2*ISS**2*w**2*np.log(10.0)*GradDenom*RProf*PAmp + tau*ISS*w*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)*GradDenom*IProf*PAmp
				ImagSGrad = 2*tau**2*ISS**2*w**2*np.log(10.0)*GradDenom*IProf*PAmp - tau*ISS*w*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)*GradDenom*RProf*PAmp

				#oldSTD = np.dot(s[i], s[i])

				s[i][:self.NFBasis] = RConfProf
				s[i][self.NFBasis:] = IConfProf

				#newSTD = np.dot(s[i], s[i])

				#s[i] *= oldSTD/newSTD

				RConfProf = RConv*j[i][:self.NFBasis] - IConv*j[i][self.NFBasis:]
				IConfProf = IConv*j[i][:self.NFBasis] + RConv*j[i][self.NFBasis:]

				j[i][:self.NFBasis] = RConfProf
				j[i][self.NFBasis:] = IConfProf

				#j[i] *= oldSTD/newSTD

				RBasis = (RConv*OneFBasis[i,:self.NFBasis,:].T - IConv*OneFBasis[i,self.NFBasis:,:].T).T
				IBasis = (IConv*OneFBasis[i,:self.NFBasis,:].T + RConv*OneFBasis[i,self.NFBasis:,:].T).T 

				OneFBasis[i,:self.NFBasis,:] = RBasis
				OneFBasis[i,self.NFBasis:,:] = IBasis
			
				#OneFBasis[i] *= oldSTD/newSTD

				'''
				plt.plot(np.linspace(0,2,2*self.NFBasis), RollData, color='black')
				plt.plot(np.linspace(0,2,2*self.NFBasis), ProfileAmps[i]*s[i], color='red')
				plt.xlabel('Frequency')
				plt.ylabel('Profile Amplitude')
				plt.show()

				plt.plot(np.linspace(0,2,2*self.NFBasis), RollData - ProfileAmps[i]*s[i], color='black')
				plt.xlabel('Frequency')
				plt.ylabel('Profile Amplitude')
				plt.show()

				bd = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
				bd[1:self.NFBasis+1] = RollData[:self.NFBasis] + 1j*RollData[self.NFBasis:]
				bdt = np.fft.irfft(bd)

				bm = np.zeros(len(np.fft.rfft(self.ProfileData[i]))) + 0j
				bm[1:self.NFBasis+1] =  ProfileAmps[i]*s[i][:self.NFBasis] + 1j* ProfileAmps[i]*s[i][self.NFBasis:]
				bmt = np.fft.irfft(bm)

				plt.plot(np.linspace(0,1,self.Nbins[i]), bdt, color='black')
				plt.plot(np.linspace(0,1,self.Nbins[i]), bmt, color='red')
				plt.xlabel('Phase')
				plt.ylabel('Profile Amplitude')
				plt.show()
				plt.plot(np.linspace(0,1,self.Nbins[i]),bdt-bmt)
				plt.xlabel('Phase')
				plt.ylabel('Profile Residuals')
				plt.show()
				'''

			Res = RollData-s[i]*ProfileAmps[i]
			Chisq = np.dot(Res,Res)/ProfileNoise[i]

			AmpGrad = -1*np.dot(s[i], Res)/ProfileNoise[i]
			NoiseGrad = (-Chisq+2*self.NFBasis)/np.sqrt(ProfileNoise[i])

			proflike = 0.5*Chisq + 0.5*2*self.NFBasis*np.log(ProfileNoise[i])


			like += proflike   

			pcount = 0
			if(self.fitPAmps == True):
				grad[pcount + i] = AmpGrad
				pcount += self.NToAs
		
			if(self.fitPNoise == True):
				grad[pcount + i] = NoiseGrad
				pcount += self.NToAs

	
			PhaseGrad = np.dot(Res, j[i])*ProfileAmps[i]/ProfileNoise[i]
			#Gradient for Phase
			if(self.fitPhase == True):	
				grad[pcount] += PhaseGrad*self.ReferencePeriod
				pcount += 1

			#Gradient for Timing Model
			if(self.numTime>0):
				TimeGrad = self.designMatrix[i]*PhaseGrad
				grad[pcount:pcount+self.numTime] += TimeGrad
				pcount += self.numTime

			#Gradient for Shape Parameters
			ShapeGrad = np.dot(OneFBasis[i].T, Res)/ProfileNoise[i]
			fvals = ((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)

			for c in range(1, self.TotCoeff):
				for p in range(self.EvoNPoly+1):
					grad[pcount] += -fvals[p]*ShapeGrad[c]*ProfileAmps[i]
					pcount += 1
	
			#Gradient for Scattering Parameters		
			if(self.fitScatter == True):
				for c in range(self.NScatterEpochs):
					if(c in self.ScatterInfo[i]):
						RScatterGrad = np.dot(RealSGrad, Res[:self.NFBasis])
						IScatterGrad = np.dot(ImagSGrad, Res[self.NFBasis:])
						grad[pcount+c] += (RScatterGrad + IScatterGrad)/ProfileNoise[i]
						#print i, (RScatterGrad + IScatterGrad)/ProfileNoise[i]
				pcount += self.NScatterEpochs

		#print (like[0]+phasePrior)[0], (Phase-self.MeanPhase)[0], grad[DiagParams], phasePriorGrad[0], (grad[DiagParams] + phasePriorGrad)[0]
		#Add phase prior to likelihood and gradient
		if(self.fitPhase == True):	
			like[0] += phasePrior
			grad[DiagParams] += phasePriorGrad


	
		#Send relevant gradients to principle coordinates for sampling

		DenseGrad = copy.copy(grad[DiagParams:])
		PrincipleGrad = np.dot(self.EigM.T, DenseGrad)
		grad[DiagParams:] = PrincipleGrad
		
		#DenseGrad = copy.copy(grad[:])
		#PrincipleGrad = np.dot(self.EigM.T, DenseGrad)
		#grad[:] = PrincipleGrad
		#print("like:", like[0], "grad", PrincipleGrad, DenseGrad)
		for i in range(ndim[0]):
			g[i] = grad[i]

		print "Params", p
		print "grad:", g

		return 

	
	def write_ghs_extract_with_logpostval(self, ndim, x, logpostval, grad):

		params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))

		DiagParams = self.DiagParams
		
		#Send relevant parameters to physical coordinates for likelihood
		if(DiagParams < self.n_params):
			DenseParams = params[DiagParams:]
			PhysParams = np.dot(self.EigM, DenseParams)
			params[DiagParams:] = PhysParams

                if(self.BaselineNoiseParams > 0):
                        for i in range(self.NToAs):
                                DenseParams = params[self.BLNList[i]]
                                PhysParams = np.dot(self.BLNEigM[i], DenseParams)
                                params[self.BLNList[i]] = PhysParams

		self.GHSoutfile.write(" ".join(map(lambda x: str(x), params[self.ParametersToWrite]))+" ")
	
		#for i in range(ndim[0]):
		#	self.GHSoutfile.write(str(params[i])+" ")
		
		self.GHSoutfile.write(str(logpostval[0])+"\n")      
		
		return

	def callGHS(self, resume=False, nburn = 100, nsamp = 100, feedback_int = 100, seed = -1,  max_steps = 10, dim_scale_fact = 0.4):

		if(resume == 0):
			self.GHSoutfile = open(self.root+"extract.dat", "w")
		else:
			self.GHSoutfile = open(self.root+"extract.dat", "a")

	

		if(self.useGPU == True and HaveGPUS == True):

			if(self.InitGPU == True):
				print "Initialising arrays on GPU before sampling\n"
				self.init_gpu_arrays()
				
	
			start = time.time()
			ghs.run_guided_hmc(self.GHSGPULike,
					self.write_ghs_extract_with_logpostval,
					self.n_params, 
					self.startPoint.astype(np.float64),
					(1.0/np.sqrt(self.EigV)).astype(np.float64),
					self.root,
					dim_scale_fact = dim_scale_fact,
					max_steps = max_steps,
					seed = seed,
					resume = resume,
					feedback_int = feedback_int,
					nburn = nburn,
					nsamp = nsamp,
					doMaxLike = 0)
			
			self.GHSoutfile.close()
			stop=time.time()
			print "GHS GPU run time: ", stop-start
	
		else:
			start = time.time()
                        ghs.run_guided_hmc(self.GHSCPULike,
                                        self.write_ghs_extract_with_logpostval,
                                        self.n_params,
                                        self.startPoint.astype(np.float64),
                                        (1.0/np.sqrt(self.EigV)).astype(np.float64),
                                        self.root,
                                        dim_scale_fact = dim_scale_fact,
                                        max_steps = max_steps,
                                        seed = seed,
                                        resume = resume,
                                        feedback_int = feedback_int,
                                        nburn = nburn,
                                        nsamp = nsamp,
                                        doMaxLike = 0)
                 
                        self.GHSoutfile.close()
                        stop=time.time()
                        print "GHS CPU run time: ", stop-start




	def init_gpu_arrays(self):



		####################Transfer Matrices to GPU and allocate empty ones########

		self.ProfAmps_GPU = gpuarray.empty((self.NToAs, self.TotCoeff, 1), np.float64)
		self.ProfAmps_Pointer = self.bptrs(self.ProfAmps_GPU)

		self.InterpFBasis_GPU = gpuarray.to_gpu(self.InterpFBasis)
		self.Interp_Pointers = np.array([self.InterpFBasis_GPU[i].ptr for i in range(len(self.InterpFBasis))], dtype=np.uint64)
		self.InterpPointers_GPU = gpuarray.to_gpu(self.Interp_Pointers)
		self.i_arr_gpu = gpuarray.empty(self.NToAs,  dtype=np.uint64)

		self.InterpJBasis_GPU = gpuarray.to_gpu(self.InterpJBasis)
		self.InterpJ_Pointers = np.array([self.InterpJBasis_GPU[i].ptr for i in range(len(self.InterpJBasis))], dtype=np.uint64)
		self.InterpJPointers_GPU = gpuarray.to_gpu(self.InterpJ_Pointers)
		self.JPointers_gpu = gpuarray.empty(self.NToAs,  dtype=np.uint64)

		self.gpu_ShiftedBinTimes = gpuarray.to_gpu((self.ShiftedBinTimes).astype(np.float64))
		self.gpu_NBins =  gpuarray.to_gpu((self.Nbins).astype(np.int32))
		self.gpu_SSBFreqs =  gpuarray.to_gpu((self.psr.ssbfreqs()).astype(np.float64))
		self.gpu_FoldingPeriods = gpuarray.to_gpu((self.FoldingPeriods).astype(np.float64))

		self.gpu_xS =  gpuarray.empty(self.NToAs, np.float64) 
		self.gpu_InterpBins =  gpuarray.empty(self.NToAs, np.int32) 
		self.gpu_WBTs =  gpuarray.empty(self.NToAs, np.float64) 
		self.gpu_RollBins =  gpuarray.empty(self.NToAs, np.int32) 
		



		self.Signal_GPU = gpuarray.empty((self.NToAs, 2*self.NFBasis, 1), np.float64)
		self.Signal_Pointer = self.bptrs(self.Signal_GPU)

		self.JSignal_GPU = gpuarray.empty((self.NToAs, 2*self.NFBasis, 1), np.float64)
		self.JSignal_Pointer = self.bptrs(self.JSignal_GPU)

		self.Flat_Data = np.zeros(2*self.NToAs*self.NFBasis)
		self.Freqs = np.zeros(self.NToAs*self.NFBasis)
		for i in range(self.NToAs):
			self.Flat_Data[i*self.NFBasis:(i+1)*self.NFBasis] = self.ProfileFData[i][:self.NFBasis]
			self.Flat_Data[(self.NToAs + i)*self.NFBasis:(self.NToAs + i+1)*self.NFBasis] = self.ProfileFData[i][self.NFBasis:]
		
			self.Freqs[i*self.NFBasis:(i+1)*self.NFBasis] = np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

		self.gpu_Data = gpuarray.to_gpu(np.float64(self.Flat_Data))
		self.gpu_RolledData =  gpuarray.empty(2*self.NToAs*self.NFBasis, np.float64)
		self.gpu_Freqs = gpuarray.to_gpu(np.float64(self.Freqs)) 

		self.ToA_Index = np.zeros(2*self.NToAs*self.NFBasis).astype(np.int32)
		self.Signal_Index = np.zeros(2*self.NToAs*self.NFBasis).astype(np.int32)

		for i in range(self.NToAs):
			self.ToA_Index[i*self.NFBasis:(i+1)*self.NFBasis]=i
			self.ToA_Index[(self.NToAs + i)*self.NFBasis:(self.NToAs + i+1)*self.NFBasis]=i
		
			self.Signal_Index[i*self.NFBasis:(i+1)*self.NFBasis] = np.arange(2*i*self.NFBasis,(2*i+1)*self.NFBasis)
			self.Signal_Index[(self.NToAs + i)*self.NFBasis:(self.NToAs + i+1)*self.NFBasis] = np.arange((2*i+1)*self.NFBasis,(2*i+2)*self.NFBasis)
		
		
		
		self.gpu_ToAIndex = gpuarray.to_gpu((self.ToA_Index).astype(np.int32))
		self.gpu_SignalIndex = gpuarray.to_gpu((self.Signal_Index).astype(np.int32))

		self.gpu_NResVec = gpuarray.empty((self.NToAs, 1, 2*self.NFBasis), np.float64)
		self.gpu_ResVec = gpuarray.empty((self.NToAs, 2*self.NFBasis, 1), np.float64)

		self.NResVec_pointer = self.bptrs(self.gpu_NResVec)
		self.ResVec_pointer = self.bptrs(self.gpu_ResVec)

		self.Chisqs_GPU = gpuarray.empty((self.NToAs, 1, 1), np.float64)
		self.Chisqs_Pointer = self.bptrs(self.Chisqs_GPU)

		self.AmpGrads_GPU = gpuarray.empty((self.NToAs, 1, 1), np.float64)
		self.AmpGrads_Pointer = self.bptrs(self.AmpGrads_GPU)

		self.JitterGrads_GPU = gpuarray.empty((self.NToAs, 1, 1), np.float64)
		self.JitterGrads_Pointer = self.bptrs(self.JitterGrads_GPU)

		self.ShapeGrads_GPU = gpuarray.empty((self.NToAs, 1, self.TotCoeff), np.float64)
		self.ShapeGrads_Pointer = self.bptrs(self.ShapeGrads_GPU)

		if(self.numTime > 0):
			self.DesignMatrix_GPU = gpuarray.to_gpu(np.float64(self.designMatrix))
		self.TimeSignal_GPU = gpuarray.zeros(self.NToAs, np.float64)


		self.fvals = np.zeros([self.EvoNPoly+1,self.NToAs])
		for i in range(self.EvoNPoly+1):
			self.fvals[i] = ((self.psr.ssbfreqs()/10.0**6 - self.EvoRefFreq)/1000.0)**(np.float64(i))


		if(self.incScatter == True):
			self.ScatteredBasis_GPU = gpuarray.zeros((self.NToAs, 2*self.NFBasis, self.TotCoeff), np.float64)
			self.ScatteredBasis_Pointer = self.bptrs(self.ScatteredBasis_GPU)
			self.ScatterGrads_GPU = gpuarray.empty((self.NToAs, 2*self.NFBasis, 1), np.float64)
			self.ScatterGrad_Pointer = self.bptrs(self.ScatterGrads_GPU)
			self.SumScatterGrad_GPU = gpuarray.empty((self.NToAs, 1, 1), np.float64)
			self.SumScatterGrad_Pointer = self.bptrs(self.SumScatterGrad_GPU)

		
			self.Scatter_Index = np.zeros(2*self.NToAs*self.NFBasis).astype(np.int32)

			for i in range(self.NToAs):
				SI = self.ScatterInfo[i][0]
				self.Scatter_Index[i*self.NFBasis:(i+1)*self.NFBasis] = SI
				self.Scatter_Index[(self.NToAs + i)*self.NFBasis:(self.NToAs + i+1)*self.NFBasis] = SI
		
		
			self.gpu_ScatterIndex = gpuarray.to_gpu((self.Scatter_Index).astype(np.int32))

		self.TimeJitterSignal_GPU = gpuarray.zeros(self.NToAs, np.float64)

		if(self.incBaselineNoise == True):
		
			self.BaselineNoiseAmps_GPU = gpuarray.zeros(self.NToAs, np.float64)
			self.BaselineNoiseSpecs_GPU = gpuarray.zeros(self.NToAs, np.float64)
			self.BaselineNoiseLike_GPU  = gpuarray.zeros(self.NToAs, np.float64)

		self.InitGPU = False

		return

	def bptrs(self, a):
	    """
	    Pointer array when input represents a batch of matrices.
	    """

	    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
			dtype=cublas.ctypes.c_void_p)







	def addPNoise(self, Fit = True, ML = None, write=True):

		self.incPNoise = True
		self.fitPNoise = Fit

                if(Fit == False):
                        write = False

		pstart, pstop = len(self.parameters), len(self.parameters)+self.NToAs
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NToAs
		AmIInLinear = 0

		self.ParamDict['PNoise'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, Fit, write)

		if(Fit == True):
			self.DiagParams += self.NToAs
			for i in range(self.NToAs):
				if(write==True):
					self.ParametersToWrite.append(len(self.parameters))
				
				self.parameters.append('ProfileNoise_'+str(i))
			
	
		if(ML == None):
			self.MLParameters.append(np.array([None]*self.NToAs))
		else:
			self.MLParameters.append(ML)
		
	def addPAmps(self, Fit = True, ML = None, Dense = False, write=True):
		self.incPAmps = True
		self.fitPAmps = Fit
		self.DensePAmps = Dense
		AmIInLinear = 0
		if(Dense == True and Fit == True):	
			AmIInLinear = 1

                if(Fit == False):
                        write = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+self.NToAs
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NToAs
	
		self.ParamDict['PAmps'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, Fit, write)

		if(Fit == True):
			if(self.DensePAmps == False):
				self.DiagParams += self.NToAs
			else:
				self.DenseParams += self.NToAs
				self.LinearParams += 1
			
			for i in range(self.NToAs):
				if(write==True):
					self.ParametersToWrite.append(len(self.parameters))
	
				self.parameters.append('ProfileAmps_'+str(i))
			
	
		if(ML == None):
			self.MLParameters.append(np.array([None]*self.NToAs))
		else:
			self.MLParameters.append(ML)

	def addPhase(self, Fit = True, ML = np.nan, write=True):
		self.incPhase = True
		self.fitPhase = Fit

                if(Fit == False):
                        write = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+1
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+1
		AmIInLinear = 0
		if(Fit == True):	
			AmIInLinear = 1
	
		self.ParamDict['Phase'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, Fit, write)

		if(Fit == True):	
			self.DenseParams += 1
			self.LinearParams += 1
		
			if(write==True):
				self.ParametersToWrite.append(len(self.parameters))
			
			self.parameters.append('Phase')
		#print "MeanPhase", self.MeanPhase, ML, np.array([ML])
		if(np.isnan(ML)):
			self.MLParameters.append(np.array([self.MeanPhase]))
		else:
			self.MLParameters.append(np.array([ML]))
	
	def addLinearTM(self, Fit = True, ML = np.array([]),  write=True):

		self.incLinearTM = True
		self.fitLinearTM = Fit

                if(Fit == False):
                        write = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+self.numTime
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.numTime
		AmIInLinear = 0
		if(Fit == True):	
			AmIInLinear = 1
	
		self.ParamDict['LinearTM'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, Fit, write)

		if(Fit == True):	
			self.DenseParams += self.numTime
			self.LinearParams += self.numTime
			for i in range(self.numTime):
				if(write==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append(self.psr.pars()[i])

		if(len(ML) == 0):
			ML = np.zeros(self.numTime)

		self.MLParameters.append(ML)
	
	def addProfile(self, Fit = True, ML = np.array([]), write=True):

		self.incProfile = True
		self.fitProfile = Fit

		if(Fit == False):
			write = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+(self.TotCoeff-1)*(self.EvoNPoly+1)
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+(self.TotCoeff-1)*(self.EvoNPoly+1)
		AmIInLinear = 0
		if(Fit == True):	
			AmIInLinear = 1
	
		self.ParamDict['Profile'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, Fit, write)

		if(Fit == True):	
			self.DenseParams += (self.TotCoeff-1)*(self.EvoNPoly+1)
			self.LinearParams += (self.TotCoeff-1)*(self.EvoNPoly+1)
			for i in range(self.TotCoeff-1):
				for j in range(self.EvoNPoly+1):
					if(write==True):
						self.ParametersToWrite.append(len(self.parameters))
					self.parameters.append('S'+str(i+1)+'E'+str(j))

		if(len(ML) == 0):
			ML = self.MLShapeCoeff

	
		self.MLParameters.append(ML)

	def addScatter(self, FitScatter = True, FitFreqScale = False, MLScatter = None, MLFreqScale = None, mode='parfile', writeScatter = True, writeFreqScale = True, RefFreq = 1, Prior = 0, StepSize = 0):

		self.incScatter = True
		self.ScatterRefFreq = RefFreq*10.0**9
		self.fitScatter = FitScatter
		self.fitScatterPrior = Prior
		self.fitScatterStepSize = StepSize
		self.ScatterInfo = self.GetScatteringParams(mode = mode)

                if(FitScatter == False):
                        writeScatter = False

                if(FitFreqScale == False):
                        writeFreqScale = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+ self.NScatterEpochs
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+ self.NScatterEpochs
		AmIInLinear = 0
	
		self.ParamDict['Scattering'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitScatter, writeScatter)

	
		if(FitScatter == True):	
			self.DenseParams += self.NScatterEpochs
			for i in range(self.NScatterEpochs):
				if(writeScatter == True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("Scatter_"+str(i))
	
		if(MLScatter == None):
			self.MLParameters.append(np.array([self.MeanScatter]*self.NScatterEpochs))
		else:
			self.MLParameters.append(MLScatter)
		
		
		#########Now add FreqScale parameter if necessary############
	
		self.fitScatterFreqScale = FitFreqScale
	
		pstart, pstop = len(self.parameters), len(self.parameters) + 1
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite) + 1
		AmIInLinear = 0
	
		self.ParamDict['ScatterFreqScale'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitFreqScale, writeFreqScale)

	
		if(FitFreqScale == True):	
			self.DenseParams += 1
			if(writeFreqScale == True):
				self.ParametersToWrite.append(len(self.parameters))
			self.parameters.append("ScatterFreqScale")
	
		if(MLFreqScale == None):
			self.MLParameters.append(np.array([4.0]))
		else:
			self.MLParameters.append(np.array([MLFreqScale]))
	
	

	def addEQUAD(self, FitSignal = True, FitPrior = True, MLSignal = None, MLPrior = None, mode='flag', flag = 'sys', model = None, Dense = None, 
		     writeSignal=True, writePrior=True):

		self.incEQUAD = True
		self.fitEQUADSignal = FitSignal
		self.fitEQUADPrior = FitPrior
		self.EQUADModel = model
	
		self.EQUADInfo = self.GetEQUADParams(mode = mode, flag=flag)

                if(FitSignal == False):
                        writeSignal = False

                if(FitPrior == False):
                        writePrior = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+self.NToAs 
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NToAs
		AmIInLinear = 0
		if(FitSignal == True):	
			AmIInLinear = 1
	
		self.ParamDict['EQUADSignal'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitSignal, writeSignal)
	
		if(FitSignal == True):	
			self.DenseParams += self.NToAs
			self.LinearParams += 1
			for i in range(self.NToAs):
				if(writeSignal==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("EQSignal_"+str(i))
				
		if(MLSignal == None):
			self.MLParameters.append(np.zeros(self.NToAs))
		else:
			self.MLParameters.append(MLSignal)
				
		pstart, pstop = len(self.parameters), len(self.parameters)+self.NumEQPriors
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NumEQPriors
		AmIInLinear = 0
	
		self.ParamDict['EQUADPrior'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitPrior, writePrior)

				
		if(FitPrior == True):	
			self.DenseParams += self.NumEQPriors
			for i in range(self.NumEQPriors):
				if(writePrior==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("EQPrior_"+str(i))
		
		if(MLPrior == None):
			self.MLParameters.append(np.ones(self.NumEQPriors)*-6)
		else:
			self.MLParameters.append(MLPrior)
		
		return
		
	def GetEQUADParams(self, mode='flag', flag = 'sys'):

		EQParamList = np.zeros(self.NToAs)

		if(mode == 'flag'):
			EQflags = np.unique(self.psr.flagvals(flag))
			self.NumEQPriors = len(EQflags)
			for i in range(self.NumEQPriors):
				select_indices = np.where(self.psr.flagvals(flag) ==  EQflags[i])[0] 
				EQParamList[select_indices] = i
					
		if(mode == 'global'):
			self.NumEQPriors = 1
		

		return EQParamList
	
	def addECORR(self, FitSignal = True, FitPrior = True, MLSignal = None, MLPrior = None, mode='flag', flag = 'sys', model = None, Dense = None, 
		     writeSignal=True, writePrior=True):

		self.incECORR = True
		self.fitECORRSignal = FitSignal
		self.fitECORRPrior = FitPrior
		self.ECORRModel = model
	
		self.ECORRInfo = self.GetECORRParams(mode = mode, flag=flag)

		if(FitSignal == False):
			writeSignal = False

		if(FitPrior == False):
			writePrior = False
	
		pstart, pstop = len(self.parameters), len(self.parameters)+self.NumEpochs
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NumEpochs
		AmIInLinear = 0
		if(FitSignal == True):	
			AmIInLinear = 1
	
		self.ParamDict['ECORRSignal'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitSignal, writeSignal)
	
		if(FitSignal == True):	
			self.DenseParams += self.NumEpochs
			self.LinearParams += 1
			for i in range(self.NumEpochs):
				if(writeSignal==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("ECORRSignal_"+str(i))
				
		if(MLSignal == None):
			self.MLParameters.append(np.zeros(self.NumEpochs))
		else:
			self.MLParameters.append(MLSignal)
				
		pstart, pstop = len(self.parameters), len(self.parameters)+self.NumECORRPriors
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NumECORRPriors
		AmIInLinear = 0
	
		self.ParamDict['ECORRPrior'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitPrior, writePrior)

				
		if(FitPrior == True):	
			self.DenseParams += self.NumECORRPriors
			for i in range(self.NumECORRPriors):
				if(writePrior==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("ECORRPrior_"+str(i))
		
		if(MLPrior == None):
			self.MLParameters.append(np.ones(self.NumECORRPriors)*-6)
		else:
			self.MLParameters.append(MLPrior)
		
		return
		
	def GetECORRParams(self, mode='flag', flag = 'sys'):

		ECORRParamList = np.zeros(self.NumEpochs)

		if(mode == 'flag'):
			ECORRflags = np.unique(self.psr.flagvals(flag)[self.ChansPerEpoch[:,0]])
			self.NumECORRPriors = len(ECORRflags)
			for i in range(self.NumECORRPriors):
				select_indices = np.where(self.psr.flagvals(flag)[self.ChansPerEpoch[:,0]] ==  ECORRflags[i])[0] 
				ECORRParamList[select_indices] = i
				
		if(mode == 'global'):
			self.NumECORRPriors = 1
		
		self.EpochIndex = np.zeros(self.NToAs).astype(np.int)
		for i in range(self.NumEpochs):
			self.EpochIndex[self.ChansPerEpoch[i]] = i

		return ECORRParamList

	def addBaselineNoise(self, FitAmpPrior = True, FitSpecPrior = True, MLAmpPrior = None, MLSpecPrior = None, writeAmpPrior=True, writeSpecPrior=True, BaselineNoiseRefFreq = 2, BaselineNoisePrior = None):

		self.incBaselineNoise = True
		self.fitBaselineNoiseAmpPrior = FitAmpPrior
		self.fitBaselineNoiseSpecPrior = FitSpecPrior
		self.BaselineNoiseRefFreq = BaselineNoiseRefFreq

		if(BaselineNoisePrior == None):
			self.BaselineNoisePrior = np.zeros(self.NToAs)
		else:
			self.BaselineNoisePrior = BaselineNoisePrior

		self.BaselineNoiseParams = 0
			
		pstart, pstop = len(self.parameters), len(self.parameters)+self.NToAs
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NToAs
		AmIInLinear = 0

		if(FitAmpPrior == False):
			writeAmpPrior = False

		self.ParamDict['BaselineNoiseAmpPrior'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitAmpPrior, writeAmpPrior)

			
		if(FitAmpPrior == True):	
			self.DiagParams += self.NToAs
			self.BaselineNoiseParams += 1
			for i in range(self.NToAs):
				if(writeAmpPrior==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("BLNPriorA_"+str(i))

	
		if(MLAmpPrior == None):
			MLP = np.log10(self.ProfileInfo[:,6]*np.sqrt(self.Nbins))
			self.MLParameters.append(MLP)
		else:
			self.MLParameters.append(MLAmpPrior)

		pstart, pstop = len(self.parameters), len(self.parameters)+self.NToAs
		MLpos = len(self.MLParameters)
		wstart, wstop = len(self.ParametersToWrite), len(self.ParametersToWrite)+self.NToAs
		AmIInLinear = 0

		if(FitSpecPrior == False):
			writeSpecPrior = False

		self.ParamDict['BaselineNoiseSpecPrior'] = (pstart, pstop, MLpos, wstart, wstop, AmIInLinear, FitSpecPrior, writeSpecPrior)

			
		if(FitSpecPrior == True):	
			self.DiagParams += self.NToAs
			self.BaselineNoiseParams += 1
			for i in range(self.NToAs):
				if(writeSpecPrior==True):
					self.ParametersToWrite.append(len(self.parameters))
				self.parameters.append("BLNPriorS_"+str(i))
	
		if(MLSpecPrior == None):
			MLP = np.ones(self.NToAs)*4
			self.MLParameters.append(MLP)
		else:
			self.MLParameters.append(MLSpecPrior)


		self.BLNList = []
	
		if(self.BaselineNoiseParams > 0):
			for i in range(self.NToAs):
				oneP = np.zeros(self.BaselineNoiseParams)
				sp = 0
				if(self.fitBaselineNoiseAmpPrior == True):
					index=self.ParamDict['BaselineNoiseAmpPrior'][0]
					oneP[sp] = index+i
					sp += 1
				
				if(self.fitBaselineNoiseSpecPrior == True):
					index=self.ParamDict['BaselineNoiseSpecPrior'][0]
					oneP[sp] = index+i
				
				self.BLNList.append(oneP.astype(np.int))

	
		return


	def removeBaselineNoise():
		self.incBaselineNoise = False
		self.incBaselineNoise = False
		self.BaselineNoisePrior = None
		self.fitBaselineNoiseAmpPrior = False
		self.fitBaselineNoiseSpecPrior = False
		self.BaselineNoiseParams = None
		self.BaselineNoiseRefFreq = 1

	def SimArchives(self, ML, addNoise = False, outDir="./SimProfs", calcAmps=False, calcNoise=False, ASCII = True, TimeDomain = False):

		self.ReturnProfile = True
		ndims = self.n_params
		params=np.ones(self.n_params)
		params[self.ParametersToWrite] = ML
		Signal, RolledData = self.GPULike(ndims, params)
		RollBins = self.gpu_RollBins.get()
		IBins = self.gpu_InterpBins.get()
		
		SimData = []
		if(TimeDomain == True):
			SimDataTD = []
		
		import shutil as sh
		
		FNames = np.unique(self.FNames)
		
		if not os.path.exists(outDir):
			os.makedirs(outDir)
		
		if(ASCII == False):
			for i in range(len(FNames)):
				Name = FNames[i].split("/")[-1]
				sh.copy(FNames[i], outDir+"/"+Name)

		for j in range(self.NToAs):
		
			FFTData = np.zeros(len(np.fft.rfft(self.ProfileData[j]))) + 0j
			s=Signal[j,:,0]
			d = np.zeros(2*self.NFBasis)
			d[:self.NFBasis] = RolledData[j*self.NFBasis:(j+1)*self.NFBasis] 
			d[self.NFBasis:] = RolledData[(self.NToAs + j)*self.NFBasis:(self.NToAs + j+1)*self.NFBasis]
		
		
			MNM = np.dot(s, s)
			dNM = np.dot(d, s)
		
			if(calcAmps == True):
				MLAmp = dNM/MNM
			else:
				MLAmp = self.startPoint[self.ParamDict['PAmps'][0]+j]
		
			Res=d-MLAmp*s
			RR = np.dot(Res, Res)
		
			
			if(calcNoise == True):
				MLSigma =  np.std(Res)
			else:
				MLSigma = self.startPoint[self.ParamDict['PNoise'][0]+j]
		
			FFTData[1:self.NFBasis+1] = MLAmp*s[:self.NFBasis] + 1j*MLAmp*s[self.NFBasis:]
		
			if(addNoise == True):
				FFTData.real[1:-1] += np.random.normal(0,1,len(FFTData.real[1:-1]))*MLSigma
				FFTData.imag[1:-1] += np.random.normal(0,1,len(FFTData.imag[1:-1]))*MLSigma
		
		
			FFTdData = np.roll(np.fft.irfft(FFTData), -RollBins[j]) + np.mean(self.ProfileData[j])
			if(TimeDomain == True):
				TD = np.roll(np.dot(self.InterpBasis[IBins[j]], self.MLShapeCoeff[:,0]), -RollBins[j])
			
			Norm = True
			if(Norm == True):
				FFTdData -= np.min(FFTdData)
				FFTdData /= np.max(FFTdData)
				
				if(TimeDomain == True):
					TD -= np.min(TD)
					TD /= np.max(TD)		
		
		
			SimData.append(FFTdData)
			if(TimeDomain == True):
				SimDataTD.append(TD)
			
		if(ASCII == False):
			for i in range(len(FNames)):
				ToAList = np.where(self.FNames==FNames[i])[0]
		
				Name = FNames[i].split("/")[-1]
				print "about to open archive:", i
				arch=psrchive.Archive_load(outDir+"/"+Name)
		
		
				npol = arch.get_npol()
				if(npol>1):
					print "PScrunch the Archives First!"
					#return
		
		
				for ToA in ToAList:
					subIndex = np.int(self.ArchiveMap[ToA,0])
					chanIndex = np.int(self.ArchiveMap[ToA,1])
		
					prof=arch.get_Profile(subIndex,0,chanIndex)
					amps=prof.get_amps()
		
					print "about to update amps:", ToA, Name, subIndex, chanIndex
					for k in range(self.Nbins[ToA]):
						amps[k] = float(SimData[ToA][k])
						#prof.__setitem__(k,float(SimData[ToA][k]))
		
				#print "about to call update:", i, outDir+"/"+Name
				#arch.update()
				print "about to call unload:", i, outDir+"/"+Name
				arch.unload(outDir+"/"+Name)
		else:
			
			for j in range(self.NToAs):
				Name = outDir+"/"+self.FNames[j].split("/")[-1]+".ASCII"
				print j, Name, self.FNames[j]
				ASCIIProf = np.zeros([self.Nbins[j], 2])
				ASCIIProf[:,0] = np.arange(0, self.Nbins[j])
			
				if(TimeDomain == False):
					ASCIIProf[:,1] = SimData[j]
				
				if(TimeDomain == True):
					ASCIIProf[:,1] = SimDataTD[j]
				
				
				f=open(Name, 'w')
				for j in range(len(ASCIIProf)):
					f.write(str(j)+" "+str(ASCIIProf[j,1])+"\n")
				f.close()

	def ShiftSats(self, phase):

		phase = np.float128(phase*self.ReferencePeriod/24/60/60)
		newSec = self.SatSecs - phase
		newSat = np.floor(self.psr.stoas)+newSec
		self.psr.stoas[:] = newSat
		self.psr.formbats()
		self.psr.formresiduals(removemean = False)
		self.residuals = self.psr.residuals(removemean = False)
		self.BatCorrs = self.psr.batCorrs()
		
		self.ProfileStartBats =  self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0.5 + self.BatCorrs
		self.ModelBats = newSec + self.BatCorrs - self.residuals/self.SECDAY
		ProfileBinTimes = (self.ProfileStartBats - self.ModelBats)*self.SECDAY
		self.ShiftedBinTimes = np.float64(np.array(ProfileBinTimes))
		self.MeanPhase = 0


	def UpdateBats(self):

		self.psr.formbats()
		self.psr.formresiduals(removemean = False)
		self.residuals = self.psr.residuals(removemean = False)
		self.BatCorrs = self.psr.batCorrs()

		self.ProfileStartBats =  self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0.5 + self.BatCorrs
		self.ModelBats = self.SatSecs + self.BatCorrs - self.residuals/self.SECDAY
		ProfileBinTimes = (self.ProfileStartBats - self.ModelBats)*self.SECDAY
		self.ShiftedBinTimes = np.float64(np.array(ProfileBinTimes))

	def PlotDM(self, chains, plot = True, outfile = None):

		Pars = self.psr.pars(which='fit')

		DMIndex = Pars.index('DM')
		DMChains = chains[self.ParamDict['LinearTM'][3]+DMIndex]*self.TempoPriors[DMIndex,1]

		DMX = [Pars[Pars.index(i)] for i in Pars if 'DMX_' in i]
		NDMX = len(DMX)

		TIndex = self.ParamDict['LinearTM'][3]
		DMXIndex = Pars.index(DMX[0])

		DMXVals = np.zeros([NDMX, 3])

		for i in range(NDMX):
			R1 = self.psr['DMXR1'+DMX[i][-5:]].val
			R2 = self.psr['DMXR2'+DMX[i][-5:]].val

			DMXVals[i,0] = (R1+R2)/2

			ScaleMean = self.TempoPriors[DMXIndex+i,0]
			ScaleErr  = self.TempoPriors[DMXIndex+i,1]

			DMXChain = ScaleMean + chains[TIndex+DMXIndex+i]*ScaleErr + DMChains
			val = np.mean(DMXChain)
			err = np.std(DMXChain)

			#print i, val, err, ScaleMean + np.mean(chains[TIndex+DMXIndex+i])*ScaleErr, np.std(chains[TIndex+DMXIndex+i])*ScaleErr
			#val = ScaleMean + np.mean(chains[TIndex+DMXIndex+i]-)*ScaleErr
			#err = np.std(chains[TIndex+DMXIndex+i])*ScaleErr

			DMXVals[i,1] = val
			DMXVals[i,2] = err

		DMXVals[:,1] -= np.mean(DMXVals[:,1])

		if(plot == True):
			plt.errorbar(DMXVals[:,0], DMXVals[:,1], yerr=DMXVals[:,2], linestyle='None')
			plt.xlabel('MJD')
			plt.ylabel('DM Variations')
			plt.show()

		if(outfile != None):
			print "writing to ", self.root+'DMVals.dat'
			np.savetxt(self.root+'DMVals.dat', DMXVals)
			

	def PlotScatter(self, chains, ref=0, plot = True, outfile = None):



		TIndex = self.ParamDict['Scattering'][3]
		
		
		ScatterVals = np.zeros([self.NScatterEpochs, 3])
		
		ScatterChains = chains[TIndex:TIndex+self.NScatterEpochs]
		MeanScatter = np.mean(10.0**ScatterChains[ref])
		ScatterChains = (10.0**ScatterChains - 10.0**ScatterChains[ref])
		
		
		vals =  np.mean(ScatterChains, axis=1)
		errs = np.std(ScatterChains, axis=1)
		
		
		Pars = self.psr.pars(which='set')
		SX = [Pars[Pars.index(i)] for i in Pars if 'SX_' in i]
		
		for i in range(self.NScatterEpochs):
		
			R1 = self.psr['SXR1'+SX[i][-5:]].val
			R2 = self.psr['SXR2'+SX[i][-5:]].val
			
			ScatterVals[i,0] = (R1+R2)/2
			
		
			ScatterVals[i,1] = vals[i]
			ScatterVals[i,2] = errs[i]
			
		#ScatterVals[:,1] -= np.mean(ScatterVals[:,1])
		ScatterVals[:,1] += MeanScatter
		
		if(plot == True):
			plt.errorbar(ScatterVals[:,0], ScatterVals[:,1], yerr=ScatterVals[:,2], linestyle='None')
			plt.xlabel('MJD')
			plt.ylabel('Scattering Variations')
			plt.show()
		
		if(outfile != None):
			print "writing to:", self.root+'ScatterVals.dat'
			np.savetxt(self.root+'ScatterVals.dat', ScatterVals)



	def AddShapeCoeffs(self, NewNumCoeffs, interpTime = 1, useNFBasis = 0):

		OldNumCoeff = self.TotCoeff

		self.MaxCoeff = np.array(NewNumCoeffs)
		self.TotCoeff = np.sum(self.MaxCoeff)


		newShape = np.zeros([self.TotCoeff, 2])
		newShape[:OldNumCoeff,:] = self.MLShapeCoeff
		self.MLShapeCoeff = newShape

		self.PreComputeFFTShapelets(interpTime = interpTime, MeanBeta = self.MeanBeta, doplot=False, useNFBasis = useNFBasis)
		self.InitGPU = True


	def LBFGSlikewrap(self, x):

		AmpPrior = 0.1
		p = self.startPoint + x*np.sqrt(1.0/np.abs(self.EigV))

		l,g = self.GPULike(self.n_params, p)
		
		if(self.fitProfile == True):
				index=self.ParamDict['Profile'][0]
				ProfileAmps = p[index:index + (self.TotCoeff-1)*(self.EvoNPoly+1)][::self.EvoNPoly+1]
				
				
				l += 0.5*np.sum(ProfileAmps**2/AmpPrior**2)
		
		return l


	def LBFGSgradwrap(self, x):

		AmpPrior = 0.1
		p = self.startPoint + x*np.sqrt(1.0/np.abs(self.EigV))

		l,g = self.GPULike(self.n_params, p)
		
		g = g*np.sqrt(1.0/np.abs(self.EigV))

		if(self.fitProfile == True):
			index=self.ParamDict['Profile'][0]
			ProfileAmps = p[index:index + (self.TotCoeff-1)*(self.EvoNPoly+1)][::self.EvoNPoly+1]
			ProfileScale = (np.sqrt(1.0/np.abs(self.EigV))[index:index + (self.TotCoeff-1)*(self.EvoNPoly+1)])[::self.EvoNPoly+1]
					
			AmpGrad = (ProfileScale/AmpPrior**2)*ProfileAmps
			
			g[index:index + (self.TotCoeff-1)*(self.EvoNPoly+1)][::self.EvoNPoly+1] += AmpGrad
		
		return g

	def FindGlobalMaximum(self):

		self.FindML = True
		self.PhasePrior = 1e-0
		self.startPoint, self.EigV, self.EigM, self.hess = self.calculateGHSHessian(diagonalGHS=True)

		if(self.InitGPU == True):
			self.init_gpu_arrays()
		
		print "Computing Global Maximum\n"	
		r2=optimize.fmin_l_bfgs_b(self.LBFGSlikewrap, np.zeros(self.n_params), fprime=self.LBFGSgradwrap)

		NumML=(self.startPoint + r2[0]*np.sqrt(1.0/np.abs(self.EigV)))

		for k1 in range(len(self.ParamDict.keys())):
			key1 = self.ParamDict.keys()[k1]
			if(self.ParamDict[key1][6] == True):

				Np1 = self.ParamDict[key1][1] - self.ParamDict[key1][0]
				index1 = self.ParamDict[key1][0]
				print "Updating:", k1, key1
				self.MLParameters[self.ParamDict[key1][2]] = NumML[index1:index1+Np1]
		
		self.FindML = False

	def UpdateStartPointFromChains(self, root = None, burnin=0):

		if(root == None):
			root = self.root
			
		chains=np.loadtxt(root).T
		ML = chains.T[burnin:][np.argmax(chains[-1][burnin:])][:-1]
		
		for k1 in range(len(self.ParamDict.keys())):
			key1 = self.ParamDict.keys()[k1]
			if(self.ParamDict[key1][7] == True):
		
				Np1 = self.ParamDict[key1][1] - self.ParamDict[key1][0]
				index1 = self.ParamDict[key1][0]
				index2 = self.ParamDict[key1][3]
				print "Updating from chains:", k1, key1, index1, index2
				self.MLParameters[self.ParamDict[key1][2]] = ML[index2:index2+Np1]	
