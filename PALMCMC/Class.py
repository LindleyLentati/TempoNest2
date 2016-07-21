from libstempo.libstempo import *
import libstempo as T
import psrchive
import numpy as np
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
import scipy as sp
import corner
import pymultinest
import math
import os
import threading, subprocess

class Likelihood(object):
    
	def __init__(self):
	
		
		self.SECDAY = 24*60*60

		self.parfile = None
		self.timfile = None
		self.psr = None  
		self.SatSecs = None
		self.SatDays = None
		self.FNames = None
		self.NToAs = None
		self.numTime = None	   
		self.TempoPriors = None

		self.ProfileData= None
		self.ProfileMJDs= None
		self.ProfileInfo= None

		self.toas= None
		self.residuals =  None
		self.BatCorrs =  None
		self.ModelBats =  None

		self.designMatrix = None
		self.FisherU = None

		self.TScrunched = None
		self.TScrunchedNoise = None

		self.Nbins = None
		self.ShiftedBinTimes = None
		self.ReferencePeriod = None
		self.ProfileStartBats = None
		self.ProfileEndBats = None

		self.MaxCoeff = None
		self.MLShapeCoeff = None
		self.MeanBeta = None
		self.MeanPhase = None

		self.doplot = None
	
		self.parameters = None
		self.pmin = None
		self.pmax = None
		self.startPoint = None
		self.cov_diag = None
		self.hess = None

		self.InterpolatedTime = None
		self.InterpBasis = None
		self.InterpJitterMatrix = None

		self.getShapeletStepSize = False
		self.TScrunchChans = None

		self.EvoRefFreq = 1400.0
		self.EvoNPoly = 0
		self.TScrunchShapeErr = None

		self.chains = None
		self.ShapePhaseCov = None


		#Model Parameters

		self.fitNCoeff = False
		self.fitNComps = False

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


	def loadPulsar(self, parfile, timfile):
		self.psr = T.tempopulsar(parfile=parfile, timfile = timfile)    
		self.psr.fit()
		self.SatSecs = self.psr.satSec()
		self.SatDays = self.psr.satDay()
		self.FNames = self.psr.fnames()
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
			subint=arch.get_Integration(i)
		
			nbins = subint.get_nbin()
			nchans = subint.get_nchan()
			npols = subint.get_npol()
			foldingperiod = subint.get_folding_period()
			inttime = subint.get_duration()
			centerfreq = subint.get_centre_frequency()
		
			#print "Subint Info:", i, nbins, nchans, npols, foldingperiod, inttime, centerfreq
		
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
				    #print "ChanInfo:", j, chanfreq, toafreq, np.sum(profamps)
				    profcount += 1
				    if(profcount == self.NToAs):
				        break



		self.ProfileInfo=np.array(self.ProfileInfo)
		self.ProfileData=np.array(self.ProfileData)

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

		self.ProfileStartBats = self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*0 + self.ProfileInfo[:,3]*0.5 + self.BatCorrs
		self.ProfileEndBats =  self.ProfileInfo[:,2]/self.SECDAY + self.ProfileInfo[:,3]*(self.ProfileInfo[:,4]-1) + self.ProfileInfo[:,3]*0.5 + self.BatCorrs

		self.Nbins = self.ProfileInfo[:,4]
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



	def TScrunch(self, doplot=True, channels=1):


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

		chanindices=np.array(chanindices)
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
		

		if(doplot == True):
			for i in range(channels):
				plt.plot(np.linspace(0,1,len(TScrunched[i])), TScrunched[i])
				plt.xlabel('Phase')
				plt.ylabel('Channel '+str(i)+' Amp')
				plt.show()	

		self.TScrunched = TScrunched
		self.TScrunchedNoise  = np.zeros(channels)
		for i in range(channels):
			self.TScrunchedNoise[i] = self.GetProfNoise(TScrunched[i])


	def FitEvoCoeffs(self, RFreq = 1400, polyorder = 1, doplot = False):

		self.EvoRefFreq = RFreq
		self.EvoNPoly = polyorder

		coeffs=self.MLShapeCoeff
		coeffs=np.array(coeffs).T

		Ncoeff=self.MaxCoeff

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

		averageFreqs=[]
		for i in range(len(chanindices)-1):
			sub = zipped[np.logical_and(zipped[:,0] < chanindices[i+1], zipped[:,0] >= chanindices[i])]
			averageFreqs.append(np.sum(sub[:,0]*sub[:,1])/np.sum(sub[:,1]))

		averageFreqs=np.array(averageFreqs)
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



	def getInitialParams(self, MaxCoeff = 1, fitNComps = 1, RFreq = 1400, polyorder = 0, parameters = None, pmin = None, pmax = None, x0 = None, cov_diag = None, burnin = 1000, outDir = './Initchains/', sampler = 'pal', resume=False):
	

		

		self.MaxCoeff = MaxCoeff
		self.fitNComps = fitNComps


		if(parameters == None):
			parameters=[]
			for i in range(self.fitNComps):
				parameters.append('Phase_'+str(i))
			for i in range(self.fitNComps):
				parameters.append('Log10_Width_'+str(i))
			for i in range(self.fitNComps):
				parameters.append('NCoeff_'+str(i))

		print "\nGetting initial fit to profile using averaged data, fitting for: ", parameters
		n_params = len(parameters)


		if(pmin == None):
			pmin=[]
			for i in range(self.fitNComps):
				pmin.append(-0.5)
			for i in range(self.fitNComps):
				pmin.append(-3.5)
			for i in range(self.fitNComps):
				pmin.append(2)

		if(pmax == None):
			pmax=[]
			for i in range(self.fitNComps):
				pmax.append(0.5)
			for i in range(self.fitNComps):
				pmax.append(0)
			for i in range(self.fitNComps):
				pmax.append(MaxCoeff)

		if(x0 == None):
			x0=[]
			for i in range(self.fitNComps):
				x0.append(0.0)
			for i in range(self.fitNComps):
				x0.append(-2)
			for i in range(self.fitNComps):
				x0.append(50)

		if(cov_diag == None):
			cov_diag=[]
			for i in range(self.fitNComps):
				cov_diag.append(0.1)
			for i in range(self.fitNComps):
				cov_diag.append(0.1)
			for i in range(self.fitNComps):
				cov_diag.append(5)


		self.pmin = np.array(pmin)
		self.pmax = np.array(pmax)
		x0 = np.array(x0)
		cov_diag = np.array(cov_diag)


		self.doplot = 0


		ML=[]

		if(sampler == 'pal'):
			sampler = ptmcmc.PTSampler(ndim=n_params,logl=self.InitialLogLike,logp=self.my_prior,
						    cov=np.diag(cov_diag**2),
						    outDir=outDir,
						    resume=resume)

			sampler.sample(p0=x0,Niter=10000,isave=10,burn=burnin,thin=1,neff=1000)

			chains=np.loadtxt('./Initchains/chain_1.txt').T

			self.chains = chains


			ML=chains.T[burnin:][np.argmax(chains[-3][burnin:])][:n_params]

		elif(sampler == 'multinest'):

			pymultinest.run(self.MNInitialLogLikeWrap, self.MNprior, n_params, importance_nested_sampling = False, resume = resume, verbose = True, sampling_efficiency = 'model', multimodal=False, n_live_points = 200, outputfiles_basename='./MNchains/Initial-')

			chains=np.loadtxt('./MNchains/Initial-phys_live.points').T
			ML=chains.T[np.argmax(chains[-2])][:n_params]


		self.doplot=1

		self.MaxCoeff = np.floor(ML[2]).astype(np.int)
		self.MeanBeta = 10.0**ML[1]
		self.MLShapeCoeff, self.TScrunchShapeErr = self.InitialLogLike(ML)
		


		self.TScrunchShapeErr = np.array(self.TScrunchShapeErr).T

		if(self.TScrunchChans > 1):
			self.FitEvoCoeffs(RFreq, polyorder)

		if(polyorder == 0):
			newShape = np.zeros([self.MaxCoeff, 2])
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

		FitNCoeff = np.floor(x[pcount:pcount+NComps]).astype(np.int)
		pcount += NComps


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

			self.TNothpl(FitNCoeff[comp], xVec, FullMatrix[ccount:ccount+FitNCoeff[comp]])

			FullMatrix[ccount:ccount+FitNCoeff[comp]] *= ExVec
			for i in range(FitNCoeff[comp]):
				FullMatrix[i+ccount] *= ScaleFactors[i]

			if(self.doplot == 1):
				for i in range(FitNCoeff[comp]):
					FullMatrix[i+ccount] /= np.std(FullMatrix[i+ccount])

			ccount+=FitNCoeff[comp]

		MTM = np.dot(FullMatrix, FullMatrix.T)

		Prior = 1000.0
		diag=MTM.diagonal().copy()
		diag += 1.0/Prior**2
		np.fill_diagonal(MTM, diag)
		try:
			Chol_MTM = sp.linalg.cho_factor(MTM.copy())
		except:
			return -np.inf

		if(self.doplot == 1):
			ShapeErrs = np.sqrt(np.linalg.inv(MTM.copy()).diagonal())[1:]

		loglike = 0
		MLCoeff=[]
		MLErrs = []
		for i in range(self.TScrunchChans):

			Md = np.dot(FullMatrix, self.TScrunched[i])			
			ML = sp.linalg.cho_solve(Chol_MTM, Md)

			s = np.dot(FullMatrix.T, ML)

			r = self.TScrunched - s	


			loglike  += -0.5*np.sum(r**2)/self.TScrunchedNoise[i]**2 

			if(self.doplot == 1):

			    plt.plot(np.linspace(0,1,ScrunchBins), self.TScrunched[i])
			    plt.plot(np.linspace(0,1,ScrunchBins),s)
			    plt.show()
			    plt.plot(np.linspace(0,1,ScrunchBins),self.TScrunched[i]-s)
			    plt.show()
			    zml=ML[1]
			    MLCoeff.append(ML[1:]/zml)
			    #print ShapeErrs, ML
			    MLErrs.append(ShapeErrs*self.TScrunchedNoise[i]/zml)

		if(self.doplot == 1):
		    return MLCoeff, MLErrs

		loglike -= np.sum(FitNCoeff)


		return loglike


	#Function returns matrix containing interpolated shapelet basis vectors given a time 'interpTime' in ns, and a Beta value to use.
	def PreComputeShapelets(self, interpTime = 1, MeanBeta = 0.1):


		print("Calculating Shapelet Interpolation Matrix : ", interpTime, MeanBeta);

		'''
		/////////////////////////////////////////////////////////////////////////////////////////////  
		/////////////////////////Profile Params//////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////
		'''

		InterpBins = np.max(self.Nbins)

		numtointerpolate = np.int(self.ReferencePeriod/InterpBins/interpTime/10.0**-9)+1
		InterpolatedTime = self.ReferencePeriod/InterpBins/numtointerpolate

		InterpShapeMatrix = np.zeros([numtointerpolate, InterpBins,self.MaxCoeff,])
		InterpJitterMatrix = np.zeros([numtointerpolate,InterpBins, self.MaxCoeff])

		MeanBeta = MeanBeta*self.ReferencePeriod


		interpStep = self.ReferencePeriod/InterpBins/numtointerpolate



		for t in range(numtointerpolate):
			self.update_progress(np.float64(t)/numtointerpolate)

			binpos = t*interpStep

			samplerate = self.ReferencePeriod/InterpBins
			x = np.linspace(binpos, binpos+samplerate*(InterpBins-1), InterpBins)
			x = ( x + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2
			x=x/MeanBeta
			ExVec = np.exp(-0.5*(x)**2)


			hermiteMatrix = np.zeros([self.MaxCoeff+1,InterpBins])
			JitterMatrix = np.zeros([InterpBins,self.MaxCoeff])

			self.TNothpl(self.MaxCoeff+1, x, hermiteMatrix)

			hermiteMatrix *= ExVec

			for i in range(self.MaxCoeff+1):

				hermiteMatrix[i] /= np.std(hermiteMatrix[i])


			hermiteMatrix = hermiteMatrix.T

			JitterMatrix[:,0] = (1.0/np.sqrt(2.0))*(-1.0*hermiteMatrix[:,1])/MeanBeta
			for i in range(1,self.MaxCoeff):
				JitterMatrix[:,i] = (1.0/np.sqrt(2.0))*(np.sqrt(1.0*i)*hermiteMatrix[:,i-1] - np.sqrt(1.0*(i+1))*hermiteMatrix[:,i+1])/MeanBeta

			InterpShapeMatrix[t]  = np.copy(hermiteMatrix[:,:self.MaxCoeff])
			InterpJitterMatrix[t] = np.copy(JitterMatrix)

		#InterpShapeMatrix = np.array(InterpShapeMatrix)
		#InterpJitterMatrix = np.array(InterpJitterMatrix)
		print("\nFinished Computing Interpolated Profiles")
		self.InterpBasis = InterpShapeMatrix
		self.InterpJitterMatrix = InterpJitterMatrix
		self.InterpolatedTime  = InterpolatedTime




	def getInitialPhase(self, doplot = True):
	

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
				like = self.PhaseLike(np.ones(1)*phase)
				phases.append(phase)
				likes.append(like)
				if(like > mlike):
					mlike = like
					mphase = np.ones(1)*phase

	
			minphase = mphase - stepsize
			maxphase = mphase + stepsize

		if(doplot == True):
			plt.scatter(phases, likes)
			plt.show()

		self.MeanPhase = mphase

		print "\n"

		self.getShapeletStepSize = True
		self.hess = self.PhaseLike(np.ones(1)*mphase)
		self.getShapeletStepSize = False

		#self.doplot=True
		#self.PhaseLike(ML)
		



	#@profile
	def PhaseLike(self, x):
	    

		pcount = 0
		phase = x[pcount]*self.ReferencePeriod
		pcount += 1

		loglike = 0

		stepsize=np.zeros([self.MaxCoeff - 1, self.EvoNPoly+1])

		xS = self.ShiftedBinTimes[:,0]-phase

		xS = ( xS + self.ReferencePeriod/2) % (self.ReferencePeriod ) - self.ReferencePeriod/2

		InterpBins = (xS%(self.ReferencePeriod/self.Nbins[:])/self.InterpolatedTime).astype(int)
		WBTs = xS-self.InterpolatedTime*InterpBins
		RollBins=(np.round(WBTs/(self.ReferencePeriod/self.Nbins[:]))).astype(np.int)


		#Multiply and shift out the shapelet model

		s=[np.roll(np.dot(self.InterpBasis[InterpBins[i]][:,:self.MaxCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*self.MLShapeCoeff, axis=1)), -RollBins[i]) for i in range(len(RollBins))]

		#Subtract mean and rescale


		s = [s[i] - np.sum(s[i])/self.Nbins[i] for i in range(self.NToAs)]	
		s = [s[i]/(np.dot(s[i],s[i])/self.Nbins[i]) for i in range(self.NToAs)]


		for i in range(self.NToAs):

			#Make design matrix.  Two components: baseline and profile shape.

			M=np.ones([2,self.Nbins[i]])
			M[1] = s[i]


			pnoise = self.ProfileInfo[i][6]

			MNM = np.dot(M, M.T)      
			MNM /= (pnoise*pnoise)

			#Invert design matrix. 2x2 so just do it numerically


			detMNM = MNM[0][0]*MNM[1][1] - MNM[1][0]*MNM[0][1]
			InvMNM = np.zeros([2,2])
			InvMNM[0][0] = MNM[1][1]/detMNM
			InvMNM[1][1] = MNM[0][0]/detMNM
			InvMNM[0][1] = -1*MNM[0][1]/detMNM
			InvMNM[1][0] = -1*MNM[1][0]/detMNM

			logdetMNM = np.log(detMNM)
			    
			#Now get dNM and solve for likelihood.
			    
			    
			dNM = np.dot(self.ProfileData[i], M.T)/(pnoise*pnoise)


			dNMMNM = np.dot(dNM.T, InvMNM)
			MarginLike = np.dot(dNMMNM, dNM)

			profilelike = -0.5*(logdetMNM - MarginLike)
			loglike += profilelike


			if(self.getShapeletStepSize == True):
				amp = dNMMNM[1]
				for j in range(self.MaxCoeff - 1):
					EvoFac = (((self.psr.freqs[i]-self.EvoRefFreq)/1000)**np.arange(0,self.EvoNPoly+1))**2
					BVec = amp*np.roll(self.InterpBasis[InterpBins[i]][:,j], -RollBins[i])
					stepsize[j] += EvoFac*np.dot(BVec, BVec)/self.ProfileInfo[i][6]/self.ProfileInfo[i][6]


			if(self.doplot == True):
			    baseline=dNMMNM[0]
			    amp = dNMMNM[1]
			    noise = np.std(self.ProfileData[i] - baseline - amp*s[i])
			    print i, amp, baseline, noise
			    plt.plot(np.linspace(0,1,self.Nbins[i]), self.ProfileData[i])
			    plt.plot(np.linspace(0,1,self.Nbins[i]),baseline+s[i]*amp)
			    plt.show()
			    plt.plot(np.linspace(0,1,self.Nbins[i]),self.ProfileData[i]-(baseline+s[i]*amp))
			    plt.show()

		if(self.getShapeletStepSize == True):
			for j in range(self.MaxCoeff - 1):
				print "step size ", j,  stepsize[j], 1.0/np.sqrt(stepsize[j])
			return 1.0/np.sqrt(stepsize)
	
		return loglike


	#@profile
	#Shifted=np.fft.irfft(np.fft.rfft(np.float64(ZeroExVec))*np.exp(1*2*np.pi*231*rfftfreqs*1j))

	def MarginLogLike(self, x):
	    

		pcount = 0
		phase=x[0]*self.ReferencePeriod#self.MeanPhase*self.ReferencePeriod
		pcount += 1

		NCoeff = self.MaxCoeff-1
		#pcount += 1


		ShapeAmps=np.zeros([self.MaxCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=x[pcount:pcount+(self.MaxCoeff-1)*(self.EvoNPoly+1)].reshape([(self.MaxCoeff-1),(self.EvoNPoly+1)])


		pcount += (self.MaxCoeff-1)*(self.EvoNPoly+1)

		TimingParameters=x[pcount:pcount+self.numTime]
		pcount += self.numTime

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

		#ShapeCoeff = np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*self.MLShapeCoeff, axis=1)

		s=[np.roll(np.dot(self.InterpBasis[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]

		#j=[np.roll(np.dot(self.InterpJitterMatrix[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]
	


		#Subtract mean and rescale


		s = [s[i] - np.sum(s[i])/self.Nbins[i] for i in range(self.NToAs)]	
		s = [s[i]/(np.dot(s[i],s[i])/self.Nbins[i]) for i in range(self.NToAs)]


		for i in range(self.NToAs):


			'''Make design matrix.  Two components: baseline and profile shape.'''

			M=np.ones([2,self.Nbins[i]])
			M[1] = s[i]


			pnoise = self.ProfileInfo[i][6]

			MNM = np.dot(M, M.T)      
			MNM /= (pnoise*pnoise)

			'''Invert design matrix. 2x2 so just do it numerically'''


			detMNM = MNM[0][0]*MNM[1][1] - MNM[1][0]*MNM[0][1]
			InvMNM = np.zeros([2,2])
			InvMNM[0][0] = MNM[1][1]/detMNM
			InvMNM[1][1] = MNM[0][0]/detMNM
			InvMNM[0][1] = -1*MNM[0][1]/detMNM
			InvMNM[1][0] = -1*MNM[1][0]/detMNM

			logdetMNM = np.log(detMNM)
			    
			'''Now get dNM and solve for likelihood.'''
			    
			    
			dNM = np.dot(self.ProfileData[i], M.T)/(pnoise*pnoise)


			dNMMNM = np.dot(dNM.T, InvMNM)
			MarginLike = np.dot(dNMMNM, dNM)

			profilelike = -0.5*(logdetMNM - MarginLike)
			loglike += profilelike

			if(self.doplot == True):
			    baseline=dNMMNM[0]
			    amp = dNMMNM[1]
			    noise = np.std(self.ProfileData[i] - baseline - amp*s)
			    print i, amp, baseline, noise
			    plt.plot(np.linspace(0,1,self.Nbins[i]), self.ProfileData[i])
			    plt.plot(np.linspace(0,1,self.Nbins[i]),baseline+s[i]*amp)
			    plt.show()
			    plt.plot(np.linspace(0,1,self.Nbins[i]),self.ProfileData[i]-(baseline+s[i]*amp))
			    plt.show()

		return loglike

	def calculateHessian(self,x):

		pcount = 0
		phase=x[0]*self.ReferencePeriod
		pcount += 1

		NCoeff = self.MaxCoeff-1
		#pcount += 1


		ShapeAmps=np.zeros([self.MaxCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=x[pcount:pcount+(self.MaxCoeff-1)*(self.EvoNPoly+1)].reshape([(self.MaxCoeff-1),(self.EvoNPoly+1)])


		pcount += (self.MaxCoeff-1)*(self.EvoNPoly+1)

		TimingParameters=x[pcount:pcount+self.numTime]
		pcount += self.numTime

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

		s=[np.roll(np.dot(self.InterpBasis[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]

		j=[np.roll(np.dot(self.InterpJitterMatrix[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]



		HessSize = 1 + self.numTime +(self.MaxCoeff-1)*(self.EvoNPoly+1)
		Hessian = np.zeros([HessSize,HessSize])

		for i in range(self.NToAs):


			smean = np.sum(s[i])/self.Nbins[i] 
			s[i] = s[i]-smean
			j[i] = j[i]-smean

			sstd = np.dot(s[i],s[i])/self.Nbins[i]
			s[i]=s[i]/np.sqrt(sstd)
			j[i]=j[i]/np.sqrt(sstd)

			M=np.ones([2,self.Nbins[i]])
			M[1] = s[i]


			pnoise = self.ProfileInfo[i][6]

			MNM = np.dot(M, M.T)      
			MNM /= (pnoise*pnoise)


			detMNM = MNM[0][0]*MNM[1][1] - MNM[1][0]*MNM[0][1]
			InvMNM = np.zeros([2,2])
			InvMNM[0][0] = MNM[1][1]/detMNM
			InvMNM[1][1] = MNM[0][0]/detMNM
			InvMNM[0][1] = -1*MNM[0][1]/detMNM
			InvMNM[1][0] = -1*MNM[1][0]/detMNM


			    
			'''Now get dNM and solve for likelihood.'''
			    
			    
			dNM = np.dot(self.ProfileData[i], M.T)/(pnoise*pnoise)

			dNMMNM = np.dot(dNM.T, InvMNM)

			baseline=dNMMNM[0]
			MLAmp = dNMMNM[1]
			MLSigma = pnoise

			HessMatrix = np.zeros([HessSize, self.Nbins[i]])


			#Phase First
			PhaseScale = MLAmp/MLSigma

			pcount = 0
			HessMatrix[pcount,:] = PhaseScale*j[i]
			pcount += 1


			#Hessian for Shapelet parameters
			fvals = ((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)

			for c in range(1, self.MaxCoeff):
				for p in range(self.EvoNPoly+1):
					HessMatrix[pcount,:] = fvals[p]*np.roll(self.InterpBasis[InterpBins[i]][:,c], -RollBins[i])*MLAmp/MLSigma
					pcount += 1

			#Hessian for Timing Model

			for c in range(1, self.numTime):
				HessMatrix[pcount,:] = j[i]*PhaseScale*self.designMatrix[i,c]
				pcount += 1


			OneHess = np.dot(HessMatrix, HessMatrix.T)
			Hessian += OneHess

		self.hess = Hessian

	def calculateShapePhaseCov(self,x):

		pcount = 0
		phase=x[0]*self.ReferencePeriod
		pcount += 1

		NCoeff = self.MaxCoeff-1
		#pcount += 1


		ShapeAmps=np.zeros([self.MaxCoeff, self.EvoNPoly+1])
		ShapeAmps[0][0] = 1
		ShapeAmps[1:]=x[pcount:pcount+(self.MaxCoeff-1)*(self.EvoNPoly+1)].reshape([(self.MaxCoeff-1),(self.EvoNPoly+1)])


		pcount += (self.MaxCoeff-1)*(self.EvoNPoly+1)

		TimingParameters=x[pcount:pcount+self.numTime]
		pcount += self.numTime

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

		s=[np.roll(np.dot(self.InterpBasis[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]

		j=[np.roll(np.dot(self.InterpJitterMatrix[InterpBins[i]][:,:NCoeff+1], np.sum(((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]



		HessSize = 1 +(self.MaxCoeff-1)
		Hessian = np.zeros([HessSize,HessSize])

		for i in range(self.NToAs):


			smean = np.sum(s[i])/self.Nbins[i] 
			s[i] = s[i]-smean
			j[i] = j[i]-smean

			sstd = np.dot(s[i],s[i])/self.Nbins[i]
			s[i]=s[i]/np.sqrt(sstd)
			j[i]=j[i]/np.sqrt(sstd)

			M=np.ones([2,self.Nbins[i]])
			M[1] = s[i]


			pnoise = self.ProfileInfo[i][6]

			MNM = np.dot(M, M.T)      
			MNM /= (pnoise*pnoise)


			detMNM = MNM[0][0]*MNM[1][1] - MNM[1][0]*MNM[0][1]
			InvMNM = np.zeros([2,2])
			InvMNM[0][0] = MNM[1][1]/detMNM
			InvMNM[1][1] = MNM[0][0]/detMNM
			InvMNM[0][1] = -1*MNM[0][1]/detMNM
			InvMNM[1][0] = -1*MNM[1][0]/detMNM


			    
			'''Now get dNM and solve for likelihood.'''
			    
			    
			dNM = np.dot(self.ProfileData[i], M.T)/(pnoise*pnoise)

			dNMMNM = np.dot(dNM.T, InvMNM)

			baseline=dNMMNM[0]
			MLAmp = dNMMNM[1]
			MLSigma = pnoise

			HessMatrix = np.zeros([HessSize, self.Nbins[i]])


			#Phase First
			PhaseScale = MLAmp/MLSigma

			pcount = 0
			HessMatrix[pcount,:] = PhaseScale*j[i]
			pcount += 1

			fvals = ((self.psr.freqs[i] - self.EvoRefFreq)/1000.0)**np.arange(0,self.EvoNPoly+1)

			for c in range(1, self.MaxCoeff):
				HessMatrix[pcount,:] = fvals[0]*np.roll(self.InterpBasis[InterpBins[i]][:,c], -RollBins[i])*MLAmp/MLSigma
				pcount += 1

			OneHess = np.dot(HessMatrix, HessMatrix.T)
			Hessian += OneHess

		self.ShapePhaseCov = Hessian
	


	#Jump proposal for the timing model parameters
	def TimeJump(self, x, iteration, beta):


		q=x.copy()
		y=np.dot(self.FisherU.T,x[-self.numTime:])
		ind = np.unique(np.random.randint(0,self.numTime,np.random.randint(0,self.numTime,1)[0]))
		ran=np.random.standard_normal(self.numTime)
		y[ind]=y[ind]+ran[ind]#/np.sqrt(FisherS[ind])

		newpars=np.dot(self.FisherU, y)
		q[-self.numTime:]=newpars

	
		return q, 0



'''
	def drawFromShapeletPrior(parameters, iter, beta):
	    
		# post-jump parameters
		q = parameters.copy()

		# transition probability
		qxy = 0

		# choose one coefficient at random to prior-draw on
		ind = np.unique(np.random.randint(1, MaxCoeff, 1))

		# where in your parameter list do the coefficients start?
		ct = 2

		for ii in ind:
		    
		    q[ct+ii] = np.random.uniform(pmin[ct+ii], pmax[ct+ii])
		    qxy += 0
	    
		return q, qxy
'''



