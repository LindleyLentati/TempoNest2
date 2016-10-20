from libstempo.libstempo import *
import libstempo as T
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
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


		self.SECDAY = 24*60*60

		self.parfile = None
		self.timfile = None
		self.root = None
		self.psr = None  
		self.SatSecs = None
		self.SatDays = None
		self.FNames = None
		self.NToAs = None
		self.numTime = None	   
		self.TempoPriors = None

		self.ProfileData = None
		self.ProfileFData = None
		self.fftFreqs = None
		self.NFBasis = None
		self.ProfileMJDs = None
		self.ProfileInfo = None

		self.toas= None
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

		self.MaxCoeff = None	
		self.TotCoeff = None
		self.MLShapeCoeff = None
		self.MeanBeta = None
		self.MeanPhase = None
		self.MeanScatter = None
		self.PhasePrior = None
		self.CompSeps = None

		self.doplot = None
	
		self.n_params = None
		self.parameters = None
		self.pmin = None
		self.pmax = None
		self.startPoint = None
		self.cov_diag = None
		self.hess = None
		self.EigM = None
		self.EigV = None
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

		#Model Parameters

		self.fitNCoeff = False
		self.fitNComps = False
		self.NScatterEpochs = 0
		self.ScatterInfo = None




		if(self.useGPU == True):

			self.CUhandle = cublas.cublasCreate()

			mod = SourceModule("""

					    #include <stdint.h>
					
					    __global__ void BinTimes(double *BinTimes, int32_t *NBins, double Phase, double *TimeSignal, double RefPeriod, double InterpTime, double *xS, int32_t  *InterpBins, double *WBTs, int32_t *RollBins, uint64_t *InterpPointers, uint64_t *SomePointers, uint64_t *InterpJPointers, uint64_t *SomeJPointers, const int32_t NToAs){

			                        const int i = blockDim.x*blockIdx.x + threadIdx.x;

						if(i < NToAs){
							xS[i] = BinTimes[i]  - Phase - TimeSignal[i] + RefPeriod*0.5; 

							xS[i] = xS[i] - trunc(xS[i]/RefPeriod)*RefPeriod; 
							xS[i] = xS[i]+RefPeriod - trunc((xS[i]+RefPeriod)/RefPeriod)*RefPeriod;
							xS[i] = xS[i] - RefPeriod*0.5;
						
							double InterpBin = xS[i] - trunc(xS[i]/(RefPeriod/NBins[i]))*(RefPeriod/NBins[i]);
							InterpBin = InterpBin + RefPeriod/NBins[i]  - trunc((InterpBin+RefPeriod/NBins[i])/(RefPeriod/NBins[i]))*(RefPeriod/NBins[i]);
							InterpBin /= InterpTime;
							InterpBins[i] = int(InterpBin);
						
							SomePointers[i] = InterpPointers[InterpBins[i]];
							
							SomeJPointers[i] = InterpJPointers[InterpBins[i]];
						
							WBTs[i] = xS[i]-InterpTime*InterpBins[i];
							RollBins[i] = int(round(WBTs[i]/(RefPeriod/NBins[i])));
						}
												
                   			}

			                    __global__ void PrepLikelihood(double *ProfAmps, double *ShapeAmps, const int32_t NToAs, const int32_t TotCoeff){
                        			
						const int i = blockDim.x*blockIdx.x + threadIdx.x;
						//double freq = ToAFreqs[i];
						
						if(i < TotCoeff*NToAs){
							
							int index = i%TotCoeff;
							double amp = ShapeAmps[index];
					
							ProfAmps[i] = amp;
							
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
 					""")
 					
			self.GPURotateData = mod.get_function("RotateData")
			self.GPUGetRes = mod.get_function("getRes")
			self.GPUBinTimes = mod.get_function("BinTimes")
			self.GPUPrepLikelihood = mod.get_function("PrepLikelihood")


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


	def loadPulsar(self, parfile, timfile, ToPickle = False, FromPickle = False, root='Example'):

		self.root = root
		self.psr = T.tempopulsar(parfile=parfile, timfile = timfile)    
		self.psr.fit()
		self.SatSecs = self.psr.satSec()
		self.SatDays = self.psr.satDay()
		self.FNames = self.psr.filename()
		self.NToAs = self.psr.nobs
		    


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

		if(FromPickle == False):
			print "Loading Data: "
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
					    profcount += 1
					    if(profcount == self.NToAs):
						break



			self.ProfileInfo=np.array(self.ProfileInfo)
			self.ProfileData=np.array(self.ProfileData)

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
		self.ModelBats = self.psr.satSec() + self.BatCorrs - self.residuals/self.SECDAY



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

		self.ProfileStartBats = self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0 + self.ProfileInfo[:,3]*0.5 + self.BatCorrs
		self.ProfileEndBats =  self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*(self.ProfileInfo[:,4]-1) + self.ProfileInfo[:,3]*0.5 + self.BatCorrs

		self.Nbins = (self.ProfileInfo[:,4]).astype(int)
		ProfileBinTimes = []
		for i in range(self.NToAs):
			ProfileBinTimes.append((np.linspace(self.ProfileStartBats[i], self.ProfileEndBats[i], self.Nbins[i])- self.ModelBats[i])*self.SECDAY)
		self.ShiftedBinTimes = np.float64(np.array(ProfileBinTimes))

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



	def TScrunch(self, doplot=True, channels=1, FromPickle = False, ToPickle = False):


		self.TScrunchChans = channels
	
		minfreq = np.min(self.psr.freqs)
		maxfreq = np.max(self.psr.freqs)

		weights = 1.0/self.psr.toaerrs**2


		zipped=np.zeros([self.NToAs,2])
		zipped[:,0]=self.psr.freqs
		zipped[:,1]=weights
		zipped=zipped[zipped[:,0].argsort()]

		totalweight=np.sum(weights)
		weightsum = np.cumsum(zipped[:,1])/totalweight

		chanindices = [minfreq]
		for i in range(channels):
			chanindices.append(zipped[(np.abs(weightsum-np.float64(i+1)/channels)).argmin()][0])

		chanindices[0] -= 1
		chanindices[-1] += 1

		chanindices=np.array(chanindices)


		averageFreqs=[]
		for i in range(len(chanindices)-1):

			sub = zipped[np.logical_and(zipped[:,0] < chanindices[i+1], zipped[:,0] >= chanindices[i])]
			averageFreqs.append(np.sum(sub[:,0]*sub[:,1])/np.sum(sub[:,1]))

		averageFreqs=np.array(averageFreqs)
		self.TScrunchedFreqs = averageFreqs

		if(FromPickle == False):
			TScrunched = np.zeros([channels,np.max(self.Nbins)])

		
			totalweight = np.zeros(channels)

			profcount = 0
			print "\nAveraging All Data In Time: "
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
					profamps = prof.get_amps()
			
					if(np.sum(profamps) != 0 and abs(toafreq-chanfreq) < 0.001):
					    noiselevel=self.GetProfNoise(profamps)
					    weight = 1.0/noiselevel**2
					    
					    try:
						value,position = min((b,a) for a,b in enumerate(chanindices-toafreq) if b>0)
					    except:
						value,position = (maxfreq, channels)
						
					    totalweight[position-1] += weight
					    TScrunched[position-1] += profamps*weight

					    profcount += 1
					    if(profcount == self.NToAs):
						break

				(TScrunched.T/totalweight).T

			for i in range(channels):
				TScrunched[i] /= np.max(TScrunched[i])

			self.TScrunched = TScrunched
			self.TScrunchedNoise  = np.zeros(channels)
			for i in range(channels):
				self.TScrunchedNoise[i] = self.GetProfNoise(TScrunched[i])

			if(ToPickle == True):
				print "\nPickling TScrunch"
                                output = open(self.root+'-TScrunch.'+str(channels)+'C.pickle', 'wb')
                                pickle.dump(self.TScrunched, output)
                                pickle.dump(self.TScrunchedNoise, output)
                                output.close()

		if(FromPickle == True):
			print "Loading TScrunch from Pickled Data"
                        pick = open(self.root+'-TScrunch.'+str(channels)+'C.pickle', 'rb')
                        self.TScrunched = pickle.load(pick)
                        self.TScrunchedNoise  = pickle.load(pick)
                        pick.close()

		if(doplot == True):
                        for i in range(channels):
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

		minfreq = np.min(self.psr.freqs)
		maxfreq = np.max(self.psr.freqs)

		weights = 1.0/self.psr.toaerrs**2


		zipped=np.zeros([self.NToAs,2])
		zipped[:,0]=self.psr.freqs
		zipped[:,1]=weights
		zipped=zipped[zipped[:,0].argsort()]

		totalweight=np.sum(weights)
		weightsum = np.cumsum(zipped[:,1])/totalweight

		chanindices = [minfreq]
		for i in range(channels):
			chanindices.append(zipped[(np.abs(weightsum-np.float64(i+1)/channels)).argmin()][0])

		chanindices=np.array(chanindices)
		refloc=np.abs(chanindices-RFreq).argmin()

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
			self.MeanScatter = ML[3*self.fitNComps+1]

		
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

		ScrunchBins = max(np.shape(self.TScrunched))
		ScrunchChans = min(np.shape(self.TScrunched))


		rfftfreqs=np.linspace(0,0.5*ScrunchBins,0.5*ScrunchBins+1)
		FullMatrix = np.ones([np.sum(FitNCoeff), len(2*np.pi*rfftfreqs)])
		FullCompMatrix = np.zeros([np.sum(FitNCoeff), len(2*np.pi*rfftfreqs)]) + 0j
		RealCompMatrix = np.zeros([np.sum(FitNCoeff), 2*len(2*np.pi*rfftfreqs)-2])

		ccount = 0
		for comp in range(NComps):


			Beta = self.ReferencePeriod*width[comp]
			if(FitNCoeff[comp] > 1):
				self.TNothpl(FitNCoeff[comp], 2*np.pi*rfftfreqs*width[comp], FullMatrix[ccount:ccount+FitNCoeff[comp]])

			ExVec = np.exp(-0.5*(2*np.pi*rfftfreqs*width[comp])**2)
			FullMatrix[ccount:ccount+FitNCoeff[comp]]=FullMatrix[ccount:ccount+FitNCoeff[comp]]*ExVec*width[comp]

			for coeff in range(FitNCoeff[comp]):
				FullCompMatrix[ccount+coeff] = FullMatrix[ccount+coeff]*(1j**coeff)

			rollVec = np.exp(2*np.pi*((phase[comp]+0.5)*ScrunchBins)*rfftfreqs/ScrunchBins*1j)

	
			ScaleFactors = self.Bconst(width[comp]*self.ReferencePeriod, np.arange(FitNCoeff[comp]))

	


			FullCompMatrix[ccount:ccount+FitNCoeff[comp]] *= rollVec
			for i in range(FitNCoeff[comp]):
				FullCompMatrix[i+ccount] *= ScaleFactors[i]


			RealCompMatrix[:,:len(2*np.pi*rfftfreqs)-1] = np.real(FullCompMatrix[:,1:len(2*np.pi*rfftfreqs)])
			RealCompMatrix[:,len(2*np.pi*rfftfreqs)-1:] = -1*np.imag(FullCompMatrix[:,1:len(2*np.pi*rfftfreqs)])

			ccount+=FitNCoeff[comp]


		if(self.NScatterEpochs == 0):

			MTM = np.dot(RealCompMatrix, RealCompMatrix.T)

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

			loglike = 0
			MLCoeff=[]
			MLErrs = []
			for i in range(self.TScrunchChans):


				FFTScrunched = np.fft.rfft(self.TScrunched[i])
				RealFFTScrunched = np.zeros(2*len(2*np.pi*rfftfreqs)-2)
				RealFFTScrunched[:len(2*np.pi*rfftfreqs)-1] = np.real(FFTScrunched[1:])
				RealFFTScrunched[len(2*np.pi*rfftfreqs)-1:] = np.imag(FFTScrunched[1:])

				Md = np.dot(RealCompMatrix, RealFFTScrunched)			
				ML = sp.linalg.cho_solve(Chol_MTM, Md)

				s = np.dot(RealCompMatrix.T, ML)

				r = RealFFTScrunched - s	
				noise  = self.TScrunchedNoise[i]*np.sqrt(ScrunchBins)/np.sqrt(2)

				loglike  += -0.5*np.sum(r**2)/noise**2  - 0.5*np.log(noise**2)*len(RealFFTScrunched)
				if(self.doplot == 1):

				    bd = np.fft.rfft(self.TScrunched[i])
				    bd[0] = 0 + 0j
				    bdt = np.fft.irfft(bd)

				    bm = np.zeros(len(np.fft.rfft(self.TScrunched[i]))) + 0j
				    bm[1:] = s[:len(s)/2] + 1j*s[len(s)/2:]
				    bmt = np.fft.irfft(bm)

				    plt.plot(np.linspace(0,1,ScrunchBins), bdt, color='black')
				    plt.plot(np.linspace(0,1,ScrunchBins), bmt, color='red')
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Amplitude')
				    plt.show()
				    plt.plot(np.linspace(0,1,ScrunchBins),bdt-bmt)
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
				ScatterVec = self.ConvolveExp(np.linspace(0, ScrunchBins/2, ScrunchBins/2+1)/self.ReferencePeriod, STime)

				ScatterMatrix = FullMatrix*ScatterVec

				RealCompMatrix[:,:len(2*np.pi*rfftfreqs)-1] = np.real(ScatterMatrix[:,1:len(2*np.pi*rfftfreqs)])
				RealCompMatrix[:,len(2*np.pi*rfftfreqs)-1:] = -1*np.imag(ScatterMatrix[:,1:len(2*np.pi*rfftfreqs)])

				

				MTM = np.dot(RealCompMatrix, RealCompMatrix.T)

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
				RealFFTScrunched = np.zeros(2*len(2*np.pi*rfftfreqs)-2)
				RealFFTScrunched[:len(2*np.pi*rfftfreqs)-1] = np.real(FFTScrunched[1:])
				RealFFTScrunched[len(2*np.pi*rfftfreqs)-1:] = np.imag(FFTScrunched[1:])				

				Md = np.dot(RealCompMatrix, RealFFTScrunched)			
				ML = sp.linalg.cho_solve(Chol_MTM, Md)

				s = np.dot(RealCompMatrix.T, ML)

				r = RealFFTScrunched - s	

				#for bin in range(1024):
				#	print i, bin, self.TScrunched[i][bin], s[bin], self.TScrunchedNoise[i]


				noise = self.TScrunchedNoise[i]*np.sqrt(ScrunchBins)/np.sqrt(2)
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

				    plt.plot(np.linspace(0,1,ScrunchBins), bdt, color='black')
				    plt.plot(np.linspace(0,1,ScrunchBins), bmt, color='red')
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Amplitude')
				    plt.show()
				    plt.plot(np.linspace(0,1,ScrunchBins),bdt-bmt)
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

	#@profile
	def InitialLogLike(self, x):


		NComps = self.fitNComps


		pcount = 0
		phase = x[pcount:pcount+NComps]


		for i in range(1, NComps):
			phase[i] = (1-phase[i-1])*phase[i]+phase[i-1]
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

		ScrunchBins = max(np.shape(self.TScrunched))
		ScrunchChans = min(np.shape(self.TScrunched))

		FullMatrix = np.ones([1+np.sum(FitNCoeff), ScrunchBins])


		ccount = 1
		for comp in range(NComps):


			xVec = (np.linspace(-0.5,0.5-1.0/ScrunchBins, ScrunchBins) - phase[comp]) 
			xVec = ( xVec + 0.5) % (1.0) - 0.5 
			xVec /= width[comp]

			ExVec = np.exp(-0.5*(xVec)**2)
			ScaleFactors = self.Bconst(width[comp], np.arange(FitNCoeff[comp]))
	
			if(FitNCoeff[comp] > 1):
				self.TNothpl(FitNCoeff[comp], xVec, FullMatrix[ccount:ccount+FitNCoeff[comp]])

			FullMatrix[ccount:ccount+FitNCoeff[comp]] *= ExVec
			for i in range(FitNCoeff[comp]):
				FullMatrix[i+ccount] *= ScaleFactors[i]

			if(self.doplot == 1):
				for i in range(FitNCoeff[comp]):
					FullMatrix[i+ccount] /= np.std(FullMatrix[i+ccount])

			ccount+=FitNCoeff[comp]



		if(self.NScatterEpochs == 0):

			MTM = np.dot(FullMatrix, FullMatrix.T)

			Prior = 1000.0
			diag=MTM.diagonal().copy()
			diag += 1.0/Prior**2
			np.fill_diagonal(MTM, diag)
			try:
				Chol_MTM = sp.linalg.cho_factor(MTM.copy())
			except:
				return -np.inf

			if(self.returnVal == 1):
				ShapeErrs = np.sqrt(np.linalg.inv(MTM.copy()).diagonal())[1:]

			loglike = 0
			MLCoeff=[]
			MLErrs = []
			for i in range(self.TScrunchChans):

				Md = np.dot(FullMatrix, self.TScrunched[i])			
				ML = sp.linalg.cho_solve(Chol_MTM, Md)

				s = np.dot(FullMatrix.T, ML)

				r = self.TScrunched[i] - s	


				loglike  += -0.5*np.sum(r**2)/self.TScrunchedNoise[i]**2  - 0.5*np.log(self.TScrunchedNoise[i]**2)*ScrunchBins
				if(self.doplot == 1):

				    plt.plot(np.linspace(0,1,ScrunchBins), self.TScrunched[i])
				    plt.plot(np.linspace(0,1,ScrunchBins),s)
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Amplitude')
				    plt.show()
				    plt.plot(np.linspace(0,1,ScrunchBins),self.TScrunched[i]-s)
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Residuals')
				    plt.show()

				if(self.returnVal == 1):
				    zml=ML[1]
				    MLCoeff.append(ML[1:]/zml)
				    MLErrs.append(ShapeErrs*self.TScrunchedNoise[i]/zml)

			if(self.returnVal == 1):
			    return MLCoeff, MLErrs

		if(self.NScatterEpochs == 1):

			loglike = 0
			MLCoeff=[]
			MLErrs = []

			FullMatrix = np.fft.rfft(FullMatrix, axis=1)
			for i in range(self.TScrunchChans):
	
				ScatterScale = (self.TScrunchedFreqs[i]*10.0**6)**4/10.0**(9.0*4.0)
				STime = STau/ScatterScale
				ScatterVec = self.ConvolveExp(np.linspace(0, ScrunchBins/2, ScrunchBins/2+1)/self.ReferencePeriod, STime)

				FFTMatrix = FullMatrix*ScatterVec
				ScatterMatrix = np.fft.irfft(FFTMatrix, axis=1)

				MTM = np.dot(ScatterMatrix, ScatterMatrix.T)

				Prior = 1000.0
				diag=MTM.diagonal().copy()
				diag += 1.0/Prior**2
				np.fill_diagonal(MTM, diag)
				try:
					Chol_MTM = sp.linalg.cho_factor(MTM.copy())
				except:
					return -np.inf

				if(self.returnVal == 1):
					ShapeErrs = np.sqrt(np.linalg.inv(MTM.copy()).diagonal())[1:]

				Md = np.dot(ScatterMatrix, self.TScrunched[i])			
				ML = sp.linalg.cho_solve(Chol_MTM, Md)

				s = np.dot(ScatterMatrix.T, ML)

				r = self.TScrunched[i] - s	

				#for bin in range(1024):
				#	print i, bin, self.TScrunched[i][bin], s[bin], self.TScrunchedNoise[i]


				chanlike  = -0.5*np.sum(r**2)/self.TScrunchedNoise[i]**2 - 0.5*np.log(self.TScrunchedNoise[i]**2)*ScrunchBins

				loglike += chanlike

				#print i, chanlike, self.TScrunchedNoise[i]
				if(self.doplot == 1):

				    plt.plot(np.linspace(0,1,ScrunchBins), self.TScrunched[i])
				    plt.plot(np.linspace(0,1,ScrunchBins),s)
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Amplitude')
				    plt.show()
				    plt.plot(np.linspace(0,1,ScrunchBins),self.TScrunched[i]-s)
				    plt.xlabel('Phase')
				    plt.ylabel('Profile Residuals')
				    plt.show()
				
				if(self.returnVal == 1):
				    zml=ML[1]
				    MLCoeff.append(ML[1:]/zml)
				    #print ShapeErrs, ML
				    MLErrs.append(ShapeErrs*self.TScrunchedNoise[i]/zml)


			if(self.returnVal == 1):
				return MLCoeff, MLErrs

		loglike -= self.TScrunchChans*np.sum(FitNCoeff)


		return loglike





	def PreComputeFFTShapelets(self, interpTime = 1, MeanBeta = 0.1, ToPickle = False, FromPickle = False, doplot = False):


		print("Calculating Shapelet Interpolation Matrix : ", interpTime, MeanBeta);

		'''
		/////////////////////////////////////////////////////////////////////////////////////////////  
		/////////////////////////Profile Params//////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////
		'''

		InterpBins = np.max(self.Nbins)

		numtointerpolate = np.int(self.ReferencePeriod/InterpBins/interpTime/10.0**-9)+1
		InterpolatedTime = self.ReferencePeriod/InterpBins/numtointerpolate
		self.InterpolatedTime  = InterpolatedTime	



		lenRFFT = len(np.fft.rfft(np.ones(InterpBins)))

		interpStep = 1.0/InterpBins/numtointerpolate


		if(FromPickle == False):


		        InterpFShapeMatrix = np.zeros([numtointerpolate, lenRFFT, np.sum(self.MaxCoeff)])+0j
		        InterpFJitterMatrix = np.zeros([numtointerpolate,lenRFFT, np.sum(self.MaxCoeff)])+0j


			for t in range(numtointerpolate):

				self.update_progress(np.float64(t)/numtointerpolate)

				binpos = t*interpStep

				rfftfreqs=np.linspace(0,0.5*InterpBins,0.5*InterpBins+1)

				ccount = 0
				for comp in range(self.fitNComps):

					OneMatrix = np.ones([self.MaxCoeff[comp]+1, len(2*np.pi*rfftfreqs)])
					OneCompMatrix = np.zeros([self.MaxCoeff[comp]+1, len(2*np.pi*rfftfreqs)]) + 0j
					OneJitterMatrix = np.zeros([self.MaxCoeff[comp]+1, len(2*np.pi*rfftfreqs)]) + 0j

					if(self.MaxCoeff[comp] > 1):
						self.TNothpl(self.MaxCoeff[comp]+1, 2*np.pi*rfftfreqs*MeanBeta[comp], OneMatrix)

					ExVec = np.exp(-0.5*(2*np.pi*rfftfreqs*MeanBeta[comp])**2)
					OneMatrix = OneMatrix*ExVec*InterpBins*np.sqrt(2*np.pi*self.MeanBeta[comp]**2)

					for coeff in range(self.MaxCoeff[comp]+1):
						OneCompMatrix[coeff] = OneMatrix[coeff]*(1j**coeff)

					rollVec = np.exp(-2*np.pi*((binpos - self.CompSeps[comp])*InterpBins)*rfftfreqs/InterpBins*1j)


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

			self.NFBasis = upperindex - 1

			self.InterpFBasis = np.zeros([numtointerpolate, self.NFBasis*2, self.TotCoeff])
			self.InterpJBasis = np.zeros([numtointerpolate, self.NFBasis*2, self.TotCoeff])
		
			self.InterpFBasis[:,:self.NFBasis,:] = InterpFShapeMatrix[:,1:upperindex].real
			self.InterpFBasis[:,self.NFBasis:,:] = InterpFShapeMatrix[:,1:upperindex].imag

			self.InterpJBasis[:,:self.NFBasis,:] = InterpFJitterMatrix[:,1:upperindex].real
			self.InterpJBasis[:,self.NFBasis:,:] = InterpFJitterMatrix[:,1:upperindex].imag

			#self.InterpFBasis = InterpFShapeMatrix[:,1:upperindex]
			#self.InterpFJitterMatrix = InterpFJitterMatrix[:,1:upperindex]

			self.InterpolatedTime  = InterpolatedTime


			Fdata =  np.fft.rfft(self.ProfileData, axis=1)[:,1:upperindex]

			self.ProfileFData = np.zeros([self.NToAs, 2*self.NFBasis])
			self.ProfileFData[:, :self.NFBasis] = np.real(Fdata)
			self.ProfileFData[:, self.NFBasis:] = np.imag(Fdata)
	
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
		    sr = np.dot(np.real(self.InterpFBasis[0]), self.MLShapeCoeff[:,0]) 
		    si = np.dot(np.imag(self.InterpFBasis[0]), self.MLShapeCoeff[:,0])
		    bm[1:self.NFBasis+1] = sr + 1j*si
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


		xS = self.ShiftedBinTimes[:,0]-phase
		xS = ( xS + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2

		InterpBins = (xS%(self.ReferencePeriod/self.Nbins[:])/self.InterpolatedTime).astype(int)
		WBTs = xS-self.InterpolatedTime*InterpBins
		RollBins=(np.round(WBTs/(self.ReferencePeriod/self.Nbins[:]))).astype(np.int)


		s = np.sum([np.dot(self.InterpFBasis[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)

		for i in range(self.NToAs):

			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

			pnoise = self.ProfileInfo[i,6]*np.sqrt(self.Nbins[i])/np.sqrt(2)

			RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
			ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RealRoll*self.ProfileFData[i][:self.NFBasis]-ImagRoll*self.ProfileFData[i][self.NFBasis:]
			RollData[self.NFBasis:] = ImagRoll*self.ProfileFData[i][:self.NFBasis]+RealRoll*self.ProfileFData[i][self.NFBasis:]
			
			

			if(self.NScatterEpochs > 0):
				ScatterScale = self.psr.ssbfreqs()[i]**4/10.0**(9.0*4.0)
				STime = (10.0**self.MeanScatter)/ScatterScale
				ScatterVec = self.ConvolveExp(rfftfreqs*self.Nbins[i]/self.ReferencePeriod, STime)

				s[i] *= ScatterVec

			FS = np.zeros(2*self.NFBasis)
			FS[:self.NFBasis] = s[i][:self.NFBasis]
			FS[self.NFBasis:] = s[i][self.NFBasis:]

			FS /= np.sqrt(np.dot(FS,FS)/(2*self.NFBasis))

			MNM = np.dot(FS, FS)/(pnoise*pnoise)
			detMNM = MNM
			logdetMNM = np.log(detMNM)

			InvMNM = 1.0/MNM

			dNM = np.dot(self.ProfileFData[i], FS)/(pnoise*pnoise)
			dNMMNM = dNM*InvMNM

			MarginLike = dNMMNM*dNM

			profilelike = -0.5*(logdetMNM - MarginLike)
			loglike += profilelike     


		return loglike



	def calculateFFTHessian(self,x):

		pcount = 0
		phase=x[0]*self.ReferencePeriod
		pcount += 1

		NCoeff = self.TotCoeff-1
		#pcount += 1


		ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=x[pcount:pcount+(NCoeff)*(self.EvoNPoly+1)].reshape([NCoeff,(self.EvoNPoly+1)])


		pcount += NCoeff*(self.EvoNPoly+1)

		TimingParameters=x[pcount:pcount+self.numTime]
		pcount += self.numTime

		ScatteringParameters = 10.0**x[pcount:pcount+self.NScatterEpochs]
		pcount += self.NScatterEpochs

		loglike = 0

		TimeSignal = np.dot(self.designMatrix, TimingParameters)

		xS = self.ShiftedBinTimes[:,0]-phase

		if(self.numTime>0):
			xS -= TimeSignal

		xS = ( xS + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2

		InterpBins = (xS%(self.ReferencePeriod/self.Nbins[:])/self.InterpolatedTime).astype(int)
		WBTs = xS-self.InterpolatedTime*InterpBins
		RollBins=(np.round(WBTs/(self.ReferencePeriod/self.Nbins[:]))).astype(np.int)


		#Multiply and shift out the shapelet model

		#s=[np.roll(np.dot(self.InterpBasis[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]

		#j=[np.roll(np.dot(self.InterpJitterMatrix[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]


		FFTS = np.sum([np.dot(self.InterpFBasis[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)
		FFTJ = np.sum([np.dot(self.InterpFJitterMatrix[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)

		HessSize = self.n_params
		Hessian = np.zeros([HessSize,HessSize])
		LinearSize = 1 + NCoeff*(self.EvoNPoly+1) + self.numTime

		for i in range(self.NToAs):

			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

			RollVec = np.exp(2*np.pi*RollBins[i]*rfftfreqs*1j)
			FFTS[i] = FFTS[i]*RollVec
			OneProf = FFTS[i]
			FFTJ[i] = FFTJ[i]*RollVec

			if(self.NScatterEpochs > 0):

				ScatterScale = self.psr.ssbfreqs()[i]**4/10.0**(9.0*4.0)
				STime = np.sum(ScatteringParameters[self.ScatterInfo[i]])/ScatterScale
				ScatterVec = self.ConvolveExp(rfftfreqs*self.Nbins[i]/self.ReferencePeriod, STime)

				OneProf = OneProf*ScatterVec
				FFTJ[i] = FFTJ[i]*ScatterVec




			FS = np.zeros(2*self.NFBasis)
			FS[:self.NFBasis] = np.real(OneProf)
			FS[self.NFBasis:] = np.imag(OneProf)

			FJ = np.zeros(2*self.NFBasis)
			FJ[:self.NFBasis] = np.real(FFTJ[i])
			FJ[self.NFBasis:] = np.imag(FFTJ[i])

			FSdot = np.sqrt(np.dot(FS,FS)/(2*self.NFBasis))

			FS /= FSdot
			FJ /= FSdot



			pnoise = self.ProfileInfo[i,6]*np.sqrt(self.Nbins[i])/np.sqrt(2)

			MNM = np.dot(FS, FS)/(pnoise*pnoise)
			InvMNM = 1.0/MNM

			dNM = np.dot(self.ProfileFData[i], FS)/(pnoise*pnoise)
			dNMMNM = dNM*InvMNM

			MLAmp = dNMMNM
			MLSigma = pnoise


			HessMatrix = np.zeros([LinearSize, 2*self.NFBasis])

			#print "ML Amp/Noise: ", i, MLAmp, MLSigma
			#Phase First
			PhaseScale = MLAmp/MLSigma

			pcount = 0
			HessMatrix[pcount,:] = PhaseScale*FJ*self.ReferencePeriod
			pcount += 1


			#Hessian for Shapelet parameters
			fvals = ((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)

			OriginalBasis = (self.InterpFBasis[InterpBins[i]].T*RollVec).T
			if(self.NScatterEpochs > 0):
				OriginalBasis = (OriginalBasis.T*ScatterVec).T

			OneBasis = np.zeros([2*self.NFBasis, self.TotCoeff])
			OneBasis[:self.NFBasis] = np.real(OriginalBasis)
			OneBasis[self.NFBasis:] = np.imag(OriginalBasis)


			for c in range(1, self.TotCoeff):
				for p in range(self.EvoNPoly+1):
					HessMatrix[pcount,:] = fvals[p]*(OneBasis[:,c]/FSdot)*MLAmp/MLSigma
					pcount += 1

			#Hessian for Timing Model

			for c in range(self.numTime):
				HessMatrix[pcount,:] = FJ*PhaseScale*self.designMatrix[i,c]
				pcount += 1


			OneHess = np.dot(HessMatrix, HessMatrix.T)
			Hessian[:LinearSize, :LinearSize] += OneHess
			'''
			for c in range(LinearSize):
				for d in range(LinearSize):
					print "Lin Hess:", c, d, OneHess[c,d]

			'''

			for c in range(self.NScatterEpochs):
				if(c in self.ScatterInfo[i]):

					tau = ScatteringParameters[c]
					f = np.linspace(1,self.NFBasis,self.NFBasis)/self.ReferencePeriod
					w = 2.0*np.pi*f
					ISS = 1.0/(self.psr.ssbfreqs()[i]**4/10.0**(9.0*4.0))

					Conv = self.ConvolveExp(f, tau*ISS)
					#MLProf = MLAmp*FFTS[i]#/FSdot
					#ConvProf = Conv*MLProf


					RConv = np.real(Conv)
					IConv = np.imag(Conv)


					#PVec = np.zeros(2*self.NFBasis)
					#PVec[:self.NFBasis] = MLAmp*np.real(FFTS[i])/FSdot
					#PVec[self.NFBasis:] = MLAmp*np.imag(FFTS[i])/FSdot

					RProf = MLAmp*np.real(FFTS[i])/FSdot
					IProf = MLAmp*np.imag(FFTS[i])/FSdot

					pnoise = self.ProfileInfo[i,6]*np.sqrt(self.Nbins[i])/np.sqrt(2)
					'''
					dval=self.ProfileFData[i][0]
					prval=PVec[:self.NFBasis][0]
					pival=PVec[self.NFBasis:][0]
					wval=w[0]
					sval=ISS
					tval=np.log10(tau)
					oval = pnoise
					dval, prval, pival, wval, sval, tval,oval

					dival=self.ProfileFData[i][self.NFBasis]
					prval=PVec[:self.NFBasis][0]
					pival=PVec[self.NFBasis:][0]
					wval=w[0]
					sval=ISS
					tval=np.log10(tau)
					oval = pnoise
					dival, prval, pival, wval, sval, tval,oval
					'''
					HessDenom = 1.0/(1.0 + tau**2*w**2*ISS**2)**3
					GradDenom = 1.0/(1.0 + tau**2*w**2*ISS**2)**2

					Reaself = (self.ProfileFData[i][:self.NFBasis] - RProf*RConv + IProf*IConv)
					RealGrad = 2*tau**2*ISS**2*w**2*np.log(10.0)*GradDenom*RProf + tau*ISS*w*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)*GradDenom*IProf
					RealHess = -(4*tau**2*ISS**2*w**2*(tau**2*ISS**2*w**2 - 1)*np.log(10.0)**2)*HessDenom*RProf - tau*ISS*w*(1+tau**2*ISS**2*w**2*(tau**2*ISS**2*w**2 - 6))*np.log(10.0)**2*HessDenom*IProf

					FullRealHess = 1*(RealHess*Reaself + RealGrad**2)*(1.0/pnoise**2)

					ImagFunc = (self.ProfileFData[i][self.NFBasis:] - RProf*IConv - IProf*RConv)
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

					
					

					Hessian[pcount+c,pcount+c] += np.sum(profhess)
					Hessian[:LinearSize, pcount+c] += -LinearScatterCross
					Hessian[pcount+c, :LinearSize] += -LinearScatterCross
					
					pcount += 1


		self.hess = Hessian



	def ConvolveExp(self, f, tau, returngrad=False):

		w = 2.0*np.pi*f
		fanalytic = 1.0/(w**2*tau**2+1) - 1j*w*tau/(w**2*tau**2+1)

		if(returngrad==False):
			return fanalytic
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

		
		return SXParamList


#def PrintEpochParams(time):
		

	def PrintEpochParams(self, time=30, string ='DMX'):

		totalinEpochs = 0
		stoas = self.psr.stoas
		mintime = stoas.min()
		maxtime = stoas.max()

		NEpochs = (maxtime-mintime)/time
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
				print string+"_000"+str(i+1)+" 0 1"
				print string+"R1_000"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_000"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_000"+str(i+1)+" 0.20435656\n"

			if(i < 99 and i >= 9):
				print string+"_00"+str(i+1)+" 0 1"
				print string+"R1_00"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_00"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_00"+str(i+1)+" 0.20435656\n"

			if(i < 999 and i >= 99):
				print string+"_0"+str(i+1)+" 0 1"
				print string+"R1_0"+str(i+1)+" "+str(Epochs[EpochList[i]])
				print string+"R2_0"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
				print string+"ER_0"+str(i+1)+" 0.20435656\n"

			if(i < 9999 and i >= 999):
				print string+"_"+str(i+1)+" 0 1"
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

		####################Cublas Parameters########################################


		alpha = np.float64(1.0)
		beta = np.float64(0.0)


		####################Copy Parameter Vector#################################

		params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))
		
		
		#Send relevant parameters to physical coordinates for likelihood
		
		DenseParams = params[2*self.NToAs:]
		PhysParams = np.dot(self.EigM, DenseParams)
		params[2*self.NToAs:] = PhysParams
		#print("Phys Params: ", PhysParams)

		grad = np.zeros(ndim[0])

		
		####################Get Parameters########################################
				

		
		pcount = 0

		ProfileAmps = params[pcount:pcount+self.NToAs]
		pcount += self.NToAs

		ProfileNoise = params[pcount:pcount+self.NToAs]*params[pcount:pcount+self.NToAs]
		pcount += self.NToAs

		gpu_Amps = gpuarray.to_gpu(np.float64(ProfileAmps))
		gpu_Noise = gpuarray.to_gpu(np.float64(ProfileNoise)) 

		Phase = params[pcount]
		phasePrior = 0.5*(Phase-self.MeanPhase)*(Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
		phasePriorGrad = (Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
		pcount += 1

		TimingParameters = params[pcount:pcount+self.numTime]
		pcount += self.numTime

		NCoeff = self.TotCoeff-1

		ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=params[pcount:pcount + NCoeff*(self.EvoNPoly+1)].reshape([NCoeff,(self.EvoNPoly+1)])

		ShapeAmps_GPU = gpuarray.to_gpu(ShapeAmps)

		pcount += NCoeff*(self.EvoNPoly+1)

		if(self.numTime > 0):
			TimingParameters_GPU = gpuarray.to_gpu(np.float64(TimingParameters))
			cublas.cublasDgemv(self.CUhandle, 't',  self.numTime, self.NToAs, alpha, self.DesignMatrix_GPU.gpudata, self.numTime, TimingParameters_GPU.gpudata, 1, beta, self.TimeSignal_GPU.gpudata, 1)


		####################Calculate Profile Amplitudes########################################

		block_size = 128
		grid_size = int(np.ceil(self.TotCoeff*self.NToAs*1.0/block_size))
		self.GPUPrepLikelihood(self.ProfAmps_GPU, ShapeAmps_GPU, np.int32(self.NToAs), np.int32(self.TotCoeff), grid=(grid_size,1), block=(block_size,1,1))


		####################Calculate Phase Offsets########################################


		block_size = 128
		grid_size = int(np.ceil(self.NToAs*1.0/block_size))
		self.GPUBinTimes(self.gpu_ShiftedBinTimes, self.gpu_NBins, np.float64(Phase*self.ReferencePeriod), self.TimeSignal_GPU, np.float64(self.ReferencePeriod), np.float64(self.InterpolatedTime), self.gpu_xS, self.gpu_InterpBins, self.gpu_WBTs, self.gpu_RollBins, self.InterpPointers_GPU, self.i_arr_gpu, self.InterpJPointers_GPU, self.JPointers_gpu, np.int32(self.NToAs),  grid=(grid_size,1), block=(block_size,1,1))


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



		####################Compute Chisq########################################

		TotBins = np.int32(2*self.NToAs*self.NFBasis)
		grid_size = int(np.ceil(self.NToAs*self.NFBasis*2.0/block_size))	


		self.GPUGetRes(self.gpu_ResVec, self.gpu_NResVec, self.gpu_RolledData, self.Signal_GPU, gpu_Amps, gpu_Noise, self.gpu_ToAIndex, self.gpu_SignalIndex, TotBins, grid=(grid_size,1), block=(block_size,1,1))

		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.ResVec_pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.Chisqs_Pointer.gpudata, 1, self.NToAs)

		ChisqVec = self.Chisqs_GPU.get()[:,0,0]
		gpu_Chisq=np.sum(ChisqVec)
		like[0] = 0.5*gpu_Chisq + 0.5*2*self.NFBasis*np.sum(np.log(ProfileNoise))


		####################Calculate Gradients for amplitude and noise levels########################################

		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.Signal_Pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.AmpGrads_Pointer.gpudata, 1, self.NToAs)
		grad[:self.NToAs] = -1*self.AmpGrads_GPU.get()[:,0,0]


		grad[self.NToAs:2*self.NToAs] = (-ChisqVec+2*self.NFBasis)/np.sqrt(ProfileNoise)

		pcount = 2*self.NToAs


		####################Calculate Gradient for Phase Offset########################################

			
		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', 1, 1, 2*self.NFBasis, alpha, self.JSignal_Pointer.gpudata, 1, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.JitterGrads_Pointer.gpudata, 1, self.NToAs)

		JGradVec = self.JitterGrads_GPU.get()[:,0,0]*ProfileAmps

		grad[pcount] = np.sum(JGradVec*self.ReferencePeriod)
		pcount += 1
		
		
		####################Calculate Gradient for Timing Model########################################
		
		
		TimeGrad = np.dot(JGradVec, self.designMatrix)
		grad[pcount:pcount+self.numTime] = TimeGrad
		pcount += self.numTime

			
		####################Calculate Gradient for Profile and Evolution########################################
		

		cublas.cublasDgemmBatched(self.CUhandle, 'n','n', self.TotCoeff, 1, 2*self.NFBasis, alpha, self.i_arr_gpu.gpudata, self.TotCoeff, self.NResVec_pointer.gpudata, 2*self.NFBasis, beta, self.ShapeGrads_Pointer.gpudata, self.TotCoeff, self.NToAs)

		ShapeGradVec = self.ShapeGrads_GPU.get()[:,0,:]

		for c in range(1, self.TotCoeff):
			for p in range(self.EvoNPoly+1):
				OneGrad = -1*np.sum(ShapeGradVec[:,c]*ProfileAmps*self.fvals[p])
				grad[pcount] = OneGrad
				pcount += 1
				

		#Add phase prior to likelihood and gradient
		like[0] += phasePrior
		grad[2*self.NToAs] += phasePriorGrad
		
					
		#Send relevant gradients to principle coordinates for sampling
		
		DenseGrad = copy.copy(grad[2*self.NToAs:])
		PrincipleGrad = np.dot(self.EigM.T, DenseGrad)
		grad[2*self.NToAs:] = PrincipleGrad
		    
		#print("like:", like[0], "grad", PrincipleGrad, DenseGrad)
		for i in range(ndim[0]):
			g[i] = grad[i]



		return 


	def GHSCPULike(self, ndim, x, like, g):
		    
		    
		params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))
		
		
		#Send relevant parameters to physical coordinates for likelihood
		
		DenseParams = params[2*self.NToAs:]
		PhysParams = np.dot(self.EigM, DenseParams)
		params[2*self.NToAs:] = PhysParams
		#print("Phys Params: ", PhysParams)
		

		grad=np.zeros(ndim[0])

		pcount = 0

		ProfileAmps = params[pcount:pcount+self.NToAs]
		pcount += self.NToAs

		ProfileNoise = params[pcount:pcount+self.NToAs]*params[pcount:pcount+self.NToAs]
		pcount += self.NToAs
		
		Phase = params[pcount]
		phasePrior = 0.5*(Phase-self.MeanPhase)*(Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
		phasePriorGrad = 1*(Phase-self.MeanPhase)/self.PhasePrior/self.PhasePrior
		pcount += 1
		
		TimingParameters = params[pcount:pcount+self.numTime]
		pcount += self.numTime
			
		NCoeff = self.TotCoeff-1
		
		ShapeAmps=np.zeros([self.TotCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=params[pcount:pcount + NCoeff*(self.EvoNPoly+1)].reshape([NCoeff,(self.EvoNPoly+1)])

		pcount += NCoeff*(self.EvoNPoly+1)
		
		
		TimeSignal = np.dot(self.designMatrix, TimingParameters)
		


		like[0] = 0

		xS = self.ShiftedBinTimes[:,0] - Phase*self.ReferencePeriod 
		
		if(self.numTime>0):
				xS -= TimeSignal
				
		xS = ( xS + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2

		InterpBins = (xS%(self.ReferencePeriod/self.Nbins[:])/self.InterpolatedTime).astype(int)
		WBTs = xS-self.InterpolatedTime*InterpBins
		RollBins=(np.round(WBTs/(self.ReferencePeriod/self.Nbins[:]))).astype(np.int)

		#ShapeAmps=self.MLShapeCoeff

		s = np.sum([np.dot(self.InterpFBasis[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)
		j = np.sum([np.dot(self.InterpJBasis[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)


		for i in range(self.NToAs):

			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]
			RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
			ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

		
			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RealRoll*self.ProfileFData[i][:self.NFBasis]-ImagRoll*self.ProfileFData[i][self.NFBasis:]
			RollData[self.NFBasis:] = ImagRoll*self.ProfileFData[i][:self.NFBasis]+RealRoll*self.ProfileFData[i][self.NFBasis:]
			
			Res = RollData-s[i]*ProfileAmps[i]
			Chisq = np.dot(Res,Res)/ProfileNoise[i]
			
			AmpGrad = -1*np.dot(s[i], Res)/ProfileNoise[i]
			NoiseGrad = (-Chisq+2*self.NFBasis)/np.sqrt(ProfileNoise[i])
			
			proflike = 0.5*Chisq + 0.5*2*self.NFBasis*np.log(ProfileNoise[i])


			like[0] += proflike   
			
			grad[i] = AmpGrad
			grad[i+self.NToAs] = NoiseGrad
			
			#Gradient for Phase
			pcount = 2*self.NToAs
			
			PhaseGrad = np.dot(Res, j[i])*ProfileAmps[i]/ProfileNoise[i]
			grad[pcount] += PhaseGrad*self.ReferencePeriod
			pcount += 1
			
			#Gradient for Timing Model
			TimeGrad = self.designMatrix[i]*PhaseGrad
			grad[pcount:pcount+self.numTime] += TimeGrad
			pcount += self.numTime
			
			#Gradient for Shape Parameters
			ShapeGrad = np.dot(self.InterpFBasis[InterpBins[i]].T, Res)/ProfileNoise[i]
			fvals = ((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)
			
			for c in range(1, self.TotCoeff):
				for p in range(self.EvoNPoly+1):
					grad[pcount] += -fvals[p]*ShapeGrad[c]*ProfileAmps[i]
					pcount += 1


		#Add phase prior to likelihood and gradient
		like[0] += phasePrior
		grad[2*self.NToAs] += phasePriorGrad
		
		
		
		#Send relevant gradients to principle coordinates for sampling
		
		DenseGrad = copy.copy(grad[2*self.NToAs:])
		PrincipleGrad = np.dot(self.EigM.T, DenseGrad)
		grad[2*self.NToAs:] = PrincipleGrad
		    
		#print("like:", like[0], "grad", PrincipleGrad, DenseGrad)
		for i in range(ndim[0]):
			g[i] = grad[i]



		return 


	def calculateGHSHessian(self, diagonalGHS = False):

		NCoeff = self.TotCoeff-1

		x0 = np.zeros(self.n_params)
		cov_diag = np.zeros(self.n_params)
		
		DenseParams = 1  + self.numTime + NCoeff*(self.EvoNPoly+1) 
		
		hess_dense = np.zeros([DenseParams,DenseParams])

		

		xS = self.ShiftedBinTimes[:,0]-self.MeanPhase*self.ReferencePeriod
		xS = ( xS + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2

		InterpBins = (xS%(self.ReferencePeriod/self.Nbins[:])/self.InterpolatedTime).astype(int)
		WBTs = xS-self.InterpolatedTime*InterpBins
		RollBins=(np.round(WBTs/(self.ReferencePeriod/self.Nbins[:]))).astype(np.int)

		ShapeAmps=self.MLShapeCoeff

		s = np.sum([np.dot(self.InterpFBasis[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)
		
		j = np.sum([np.dot(self.InterpJBasis[InterpBins], ShapeAmps[:,i])*(((self.psr.freqs - self.EvoRefFreq)/1000.0)**i).reshape(self.NToAs,1) for i in range(self.EvoNPoly+1)], axis=0)


		for i in range(self.NToAs):


			rfftfreqs=np.linspace(1,self.NFBasis,self.NFBasis)/self.Nbins[i]

			RealRoll = np.cos(-2*np.pi*RollBins[i]*rfftfreqs)
			ImagRoll = np.sin(-2*np.pi*RollBins[i]*rfftfreqs)

		
			RollData = np.zeros(2*self.NFBasis)
			RollData[:self.NFBasis] = RealRoll*self.ProfileFData[i][:self.NFBasis]-ImagRoll*self.ProfileFData[i][self.NFBasis:]
			RollData[self.NFBasis:] = ImagRoll*self.ProfileFData[i][:self.NFBasis]+RealRoll*self.ProfileFData[i][self.NFBasis:]
			

			MNM = np.dot(s[i], s[i])
			dNM = np.dot(RollData, s[i])
			MLAmp = dNM/MNM

			PSignal = MLAmp*s[i]

			Res=RollData-PSignal
			MLSigma = np.std(Res)


			x0[i] = MLAmp
			x0[i+self.NToAs] = MLSigma

			RR = np.dot(Res, Res)

			AmpStep =  MNM/(MLSigma*MLSigma)
			SigmaStep = 3*RR/(MLSigma*MLSigma*MLSigma*MLSigma) - 2.0*self.NFBasis/(MLSigma*MLSigma)


			cov_diag[i] = AmpStep
			cov_diag[i+self.NToAs] = SigmaStep
			
			#Make Matrix for Linear Parameters
			LinearSize = 1  + self.numTime + NCoeff*(self.EvoNPoly+1) 

			HessMatrix = np.zeros([LinearSize, 2*self.NFBasis])
			
			#Hessian for Phase parameter
			
			PhaseScale = -1*MLAmp/MLSigma
			LinCount = 0
			HessMatrix[LinCount,:] = PhaseScale*j[i]*self.ReferencePeriod
			LinCount += 1
			
			#Hessian for Timing Model

			for c in range(self.numTime):
				HessMatrix[LinCount, :] = j[i]*PhaseScale*self.designMatrix[i,c]
				LinCount += 1
				

			#Hessian for Shapelet parameters
			
			fvals = ((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)

			ShapeBasis = self.InterpFBasis[InterpBins[i]]

			for c in range(1, self.TotCoeff):
				for p in range(self.EvoNPoly+1):
					HessMatrix[LinCount, :] = fvals[p]*ShapeBasis[:,c]*MLAmp/MLSigma
					LinCount += 1
					
					
							
			OneHess = np.dot(HessMatrix, HessMatrix.T)
			
			#add phase prior to hessian
			OneHess[0,0]  += (1.0/self.PhasePrior/self.PhasePrior)/self.NToAs
			
			cov_diag[self.NToAs*2:] += OneHess.diagonal()
			hess_dense += OneHess
			
			

		if(diagonalGHS == False):		
			#Now do EVD on the dense part of the matrix
		
			V, M = sl.eigh(hess_dense)
		
			cov_diag[self.NToAs*2:] = V
		
		else:
		
			hess_dense = np.eye(np.shape(hess_dense)[0])
			M = copy.copy(hess_dense)
			
		
		#Complete the start point by filling in extra parameters

		pcount = self.NToAs*2
		
		x0[pcount] = self.MeanPhase
		pcount += 1
		
		x0[pcount:pcount + self.numTime] = 0
		pcount += self.numTime
		
		if(self.EvoNPoly == 0):
			x0[pcount:pcount + NCoeff*(self.EvoNPoly+1)] = (self.MLShapeCoeff[1:].T).flatten()[:self.TotCoeff-1]
		else:
			x0[pcount:pcount + NCoeff*(self.EvoNPoly+1)] = (self.MLShapeCoeff[1:]).flatten()
			
		pcount += NCoeff*(self.EvoNPoly+1)
			
				
		return x0, cov_diag, M, hess_dense
	
	def write_ghs_extract_with_logpostval(self, ndim, x, logpostval, grad):

		params = copy.copy(np.ctypeslib.as_array(x, shape=(ndim[0],)))
		
		#Send relevant parameters to physical coordinates for likelihood
		
		DenseParams = params[2*self.NToAs:]
		PhysParams = np.dot(self.EigM, DenseParams)
		params[2*self.NToAs:] = PhysParams

		for i in range(ndim[0]):
			self.GHSoutfile.write(str(params[i])+" ")
		
		self.GHSoutfile.write(str(logpostval[0])+"\n")      
		
		return

	def callGHS(self, resume=False, nburn = 100, nsamp = 100, feedback_int = 100, seed = -1,  max_steps = 10, dim_scale_fact = 0.4):

		if(resume == 0):
			self.GHSoutfile = open(self.root+"extract.dat", "w")
		else:
			self.GHSoutfile = open(self.root+"extract.dat", "a")

	

		if(self.useGPU == True and HaveGPUS == True):


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

		self.gpu_ShiftedBinTimes = gpuarray.to_gpu((self.ShiftedBinTimes[:,0]).astype(np.float64))
		self.gpu_NBins =  gpuarray.to_gpu((self.Nbins).astype(np.int32))

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


		self.DesignMatrix_GPU = gpuarray.to_gpu(np.float64(self.designMatrix))
		self.TimeSignal_GPU = gpuarray.zeros(self.NToAs, np.float64)


		self.fvals = np.zeros([self.EvoNPoly+1,self.NToAs])
		for i in range(self.EvoNPoly+1):
			self.fvals[i] = ((self.psr.freqs - self.EvoRefFreq)/1000.0)**(np.float64(i))


	def bptrs(self, a):
	    """
	    Pointer array when input represents a batch of matrices.
	    """

	    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
			dtype=cublas.ctypes.c_void_p)
