import numpy as np
import psrchive
from libstempo.libstempo import *
import libstempo as T
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc
from scipy.optimize import minimize

	

#Funtion to determine an estimate of the white noise in the profile data
def GetProfNoise(profamps):

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


#Function returns matrix containing interpolated shapelet basis vectors given a time 'interpTime' in ns, and a Beta value to use.
def PreComputeShapelets(interpTime = 1, MeanBeta = 1):


	print("Calculating Shapelet Interpolation Matrix : ", interpTime, MeanBeta);

	'''
	/////////////////////////////////////////////////////////////////////////////////////////////  
	/////////////////////////Profile Params//////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////
	'''

	numtointerpolate = np.int(ReferencePeriod/1024/interpTime/10.0**-9)+1
	InterpolatedTime = ReferencePeriod/1024/numtointerpolate

	InterpShapeMatrix = []
	MeanBeta = MeanBeta/1024*ReferencePeriod

	InterpBins = 1024
	interpStep = ReferencePeriod/InterpBins/numtointerpolate
	
	

	for t in range(numtointerpolate):


		binpos = t*interpStep

		samplerate = ReferencePeriod/InterpBins
		x = np.linspace(binpos, binpos+samplerate*(InterpBins-1), InterpBins)
		x = ( x + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2
		x=x/MeanBeta

		hermiteMatrix = np.zeros([InterpBins, MaxCoeff])
		for i in range(MaxCoeff):
			amps = np.zeros(MaxCoeff)
			amps[i] = 1
			s = numpy.polynomial.hermite.hermval(x, amps)*np.exp(-0.5*(x)**2)
			hermiteMatrix[:,i] = s
		InterpShapeMatrix.append(np.copy(hermiteMatrix))

	
	InterpShapeMatrix = np.array(InterpShapeMatrix)
	print("Finished Computing Interpolated Profiles")
	return InterpShapeMatrix, InterpolatedTime





SECDAY = 24*60*60

#First load pulsar.  We need the sats (separate day/second), and the file names of the archives (FNames)
psr = T.tempopulsar(parfile="OneChan.par", timfile = "OneChan.tim")
psr.fit()
SatSecs = psr.satSec()
SatDays = psr.satDay()
FNames = psr.fnames()
NToAs = psr.nobs


#Check how many timing model parameters we are fitting for (in addition to phase)
numTime=len(psr.pars())
redChisq = psr.chisq()/(psr.nobs-len(psr.pars())-1)
TempoPriors=np.zeros([numTime,2]).astype(np.float128)
for i in range(numTime):
        TempoPriors[i][0]=psr[psr.pars()[i]].val
        TempoPriors[i][1]=psr[psr.pars()[i]].err/np.sqrt(redChisq)
	print "fitting for: ", psr.pars()[i], TempoPriors[i][0], TempoPriors[i][1]

#Now loop through archives, and work out what subint/frequency channel is associated with a ToA.
#Store whatever meta data is needed (MJD of the bins etc)
#If multiple polarisations are present we first PScrunch.

ProfileData=[]
ProfileMJDs=[]
ProfileInfo=[]


profcount = 0
while(profcount < NToAs):
    arch=psrchive.Archive_load(FNames[profcount])

    
    npol = arch.get_npol()
    if(npol>1):
        arch.pscrunch()

    nsub=arch.get_nsubint()


    for i in range(nsub):
        subint=arch.get_Integration(i)
        
        nbins = subint.get_nbin()
        nchans = subint.get_nchan()
        npols = subint.get_npol()
        foldingperiod = subint.get_folding_period()
        inttime = subint.get_duration()
        centerfreq = subint.get_centre_frequency()
        
        print "Subint Info:", i, nbins, nchans, npols, foldingperiod, inttime, centerfreq
        
        firstbin = subint.get_epoch()
        intday = firstbin.intday()
        fracday = firstbin.fracday()
        intsec = firstbin.get_secs()
        fracsecs = firstbin.get_fracsec()
        isdedispersed = subint.get_dedispersed()
        
        pulsesamplerate = foldingperiod/nbins/SECDAY;
        
        nfreq=subint.get_nchan()
        
        FirstBinSec = intsec + np.float128(fracsecs)
        SubIntTimeDiff = FirstBinSec-SatSecs[profcount]*SECDAY
        PeriodDiff = SubIntTimeDiff*psr['F0'].val
        
        if(abs(PeriodDiff) < 2.0):
            for j in range(nfreq):
                chanfreq = subint.get_centre_frequency(j)
                toafreq = psr.freqs[profcount]
                prof=subint.get_Profile(0,j)
                profamps = prof.get_amps()
                
                if(np.sum(profamps) != 0 and abs(toafreq-chanfreq) < 0.001):
		    noiselevel=GetProfNoise(profamps)
                    ProfileData.append(np.copy(profamps))


		   # plt.plot(np.linspace(0,1,1024), ProfileData[0])
		    #plt.plot(np.linspace(0,1,1024), s*(np.max(ProfileData[i])-np.min(ProfileData[i]))+np.min(ProfileData[i]))
		   # plt.show()


                    ProfileInfo.append([SatSecs[profcount], SatDays[profcount], np.float128(intsec)+np.float128(fracsecs), pulsesamplerate, nbins, foldingperiod, noiselevel])                    
                    print "ChanInfo:", j, chanfreq, toafreq, np.sum(profamps)
                    profcount += 1
                    if(profcount == NToAs):
                        break


len(ProfileData)
ProfileInfo=np.array(ProfileInfo)
ProfileData=np.array(ProfileData)

toas=psr.toas()
residuals = psr.residuals(removemean=False)
BatCorrs = psr.batCorrs()
ModelBats = psr.satSec() + BatCorrs - residuals/SECDAY



#get design matrix for linear timing model, setup jump proposal

designMatrix=psr.designmatrix(incoffset=False)
for i in range(numTime):
	designMatrix[:,i] *= TempoPriors[i][1]
	zval = designMatrix[0,i]
	designMatrix[:,i] -= zval

designMatrix=np.float64(designMatrix)
N=np.diag(1.0/(psr.toaerrs*10.0**-6))
Fisher=np.dot(designMatrix.T, np.dot(N, designMatrix))
FisherU,FisherS,FisherVT=np.linalg.svd(Fisher)



#Jump proposal for the timing model parameters
def TimeJump(x, iteration, beta):


	q=x.copy()
	y=np.dot(FisherU.T,x[-numTime:])
	ind = np.unique(np.random.randint(0,numTime,np.random.randint(0,numTime,1)[0]))
	ran=np.random.standard_normal(numTime)
	y[ind]=y[ind]+ran[ind]#/np.sqrt(FisherS[ind])

	newpars=np.dot(FisherU, y)
	q[-numTime:]=newpars

	
	return q, 0




ProfileStartBats = ProfileInfo[:,2]/SECDAY + ProfileInfo[:,3]*0 + ProfileInfo[:,3]*0.5 + BatCorrs
ProfileEndBats =  ProfileInfo[:,2]/SECDAY + ProfileInfo[:,3]*(ProfileInfo[:,4]-1) + ProfileInfo[:,3]*0.5 + BatCorrs

Nbins = ProfileInfo[:,4]
ProfileBinTimes = []
for i in range(NToAs):
	ProfileBinTimes.append((np.linspace(ProfileStartBats[i], ProfileEndBats[i], Nbins[i])-ModelBats[i])*SECDAY)
ShiftedBinTimes = np.float64(np.array(ProfileBinTimes))

ReferencePeriod = np.float64(ProfileInfo[0][5])



def my_prior(x):
    logp = 0.

    if np.all(x <= pmax) and np.all(x >= pmin):
        logp = np.sum(np.log(1/(pmax-pmin)))
    else:
        logp = -np.inf

    return logp

MaxCoeff = 10

InterpBasis, InterpolatedTime = PreComputeShapelets(interpTime = 1, MeanBeta = 47.9)


#@profile
def MarginLogLike(x):
    

	pcount = 0
	phase=3.36203222e-03 #x[pcount]
	pcount += 1

	NCoeff = MaxCoeff-1
	pcount += 1


	ShapeAmps=np.zeros(MaxCoeff)
	ShapeAmps[0] = 1
	ShapeAmps[1:] = x[pcount:pcount+(MaxCoeff-1)]
	pcount += MaxCoeff-1

	TimingParameters=x[pcount:pcount+numTime]
	pcount += numTime

	loglike = 0

	TimeSignal = np.dot(designMatrix, TimingParameters)

	for i in range(NToAs):

	

		'''Start by working out position in phase of the model arrival time'''


		x = ShiftedBinTimes[i]-phase-TimeSignal[i]
		x[0] = ( x[0] + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2

		InterpBin = np.int(x[0]%(ReferencePeriod/1024)/InterpolatedTime)
		WBT = x[0]-InterpolatedTime*InterpBin
		RollBins=np.int(np.round(WBT/(ReferencePeriod/1024)))


		'''Evaulate Shapelet model: to be replaced with interpolated matrix'''

		s = np.roll(np.dot(InterpBasis[InterpBin][:,:NCoeff+1], ShapeAmps[:NCoeff+1]), -RollBins)

		'''Now subtract mean and scale so std is one.  Makes the matrix stuff stable.'''

		smean = np.sum(s)/Nbins[i] 
		s = s-smean

		sstd = np.dot(s,s)/Nbins[i]
		s=s/np.sqrt(sstd)

		'''Make design matrix.  Two components: baseline and profile shape.'''

		M=np.ones([2,Nbins[i]])
		M[1] = s


		pnoise = ProfileInfo[i][6]

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
		    
		    
		dNM = np.dot(ProfileData[i], M.T)/(pnoise*pnoise)


		dNMMNM = np.dot(dNM.T, InvMNM)
		MarginLike = np.dot(dNMMNM, dNM)

		profilelike = -0.5*(logdetMNM - MarginLike)
		loglike += profilelike

		if(doplot == True):
		    baseline=dNMMNM[0]
		    amp = dNMMNM[1]
		    noise = np.std(ProfileData[i] - baseline - amp*s)
		    print i, amp, baseline, noise
		    plt.plot(np.linspace(0,1,Nbins[i]), ProfileData[i])
		    plt.plot(np.linspace(0,1,Nbins[i]),baseline+s*amp)
		    plt.show()
		    plt.plot(np.linspace(0,1,Nbins[i]),ProfileData[i]-(baseline+s*amp))
		    plt.show()

	return loglike




parameters = []
parameters.append('Phase')
parameters.append('NCoeff')
for i in range(MaxCoeff-1):
	parameters.append('S'+str(i))
for i in range(numTime):
	parameters.append(psr.pars()[i])



print parameters
n_params = len(parameters)
print n_params

    
pmin = np.array(np.ones(n_params))*-1024
pmax = np.array(np.ones(n_params))*1024


pmin[1] = 1
pmax[1] = MaxCoeff  #Ncoeff
pmin[2:2+MaxCoeff-1] = -1
pmax[2:2+MaxCoeff-1] = 1



x0 = np.array(np.zeros(n_params))

pcount = 0
x0[pcount] = 0.002883
pcount += 1

x0[pcount] = 9
pcount += 1


for i in range(MaxCoeff-1):
	x0[pcount+i] = 0

#x0[pcount:pcount+len(ShapeMax)] = ShapeMax
pcount += MaxCoeff-1

for i in range(numTime):
	x0[1+i] = 0
pcount += numTime




cov_diag = np.array(np.ones(n_params))

pcount = 0
cov_diag[pcount] = 0.0000115669002113
pcount += 1
cov_diag[pcount] = 1
pcount += 1
for i in range(MaxCoeff-1):
        cov_diag[pcount+i] = 0.0001
pcount += MaxCoeff-1
for i in range(numTime):
        cov_diag[1+i] = 1
pcount += numTime




xload=np.loadtxt('ML.dat')
x0[:11]=xload[:11]

covload = np.loadtxt('Cov.dat')
cov_diag[:11] = covload[:11]

doplot=False
burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=MarginLogLike,logp=my_prior,
                            cov=np.diag(cov_diag**2),
                            outDir='./chains/',
                            resume=True)
sampler.addProposalToCycle(TimeJump, 20)
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
STD=np.zeros(n_params)
for i in range(n_params):
	STD[i]  =  np.std(chains[i][burnin:])
	print "param:", i, np.mean(chains[i][burnin:]), np.std(chains[i][burnin:])
np.savetxt("Cov.dat", STD)
cov_diag = STD
x0=ML
'''



def TScrunch():

	TScrunched = np.zeros(1024)
	totalweight = 0

	profcount = 0
	while(profcount < NToAs):
	    arch=psrchive.Archive_load(FNames[profcount])

	    
	    npol = arch.get_npol()
	    if(npol>1):
		arch.pscrunch()

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
		
		pulsesamplerate = foldingperiod/nbins/SECDAY;
		
		nfreq=subint.get_nchan()
		
		FirstBinSec = intsec + np.float128(fracsecs)
		SubIntTimeDiff = FirstBinSec-SatSecs[profcount]*SECDAY
		PeriodDiff = SubIntTimeDiff*psr['F0'].val
		
		if(abs(PeriodDiff) < 2.0):
		    for j in range(nfreq):
		        chanfreq = subint.get_centre_frequency(j)
		        toafreq = psr.freqs[profcount]
		        prof=subint.get_Profile(0,j)
		        profamps = prof.get_amps()
		        
		        if(np.sum(profamps) != 0 and abs(toafreq-chanfreq) < 0.001):
			    noiselevel=GetProfNoise(profamps)
			    weight = 1.0/noiselevel**2
			    totalweight += weight

			    TScrunched += profamps*weight

		            profcount += 1
		            if(profcount == NToAs):
		                break

	TScrunched /= totalweight

	plt.plot(np.linspace(0,1,1024), TScrunched)
	plt.show()	

	return TScrunched


TScrunched = TScrunch()
TSNoise = GetProfNoise(TScrunched)

#@profile
def InitialLogLike(x):
    

	pcount = 0
	phase = x[pcount]
	pcount += 1
	width = x[pcount]
	pcount += 1


	loglike = 0


	'''Start by working out position in phase of the model arrival time'''


	x = (np.linspace(0, 1, 1024) - phase)/width

	hermiteMatrix = np.zeros([1024, MaxCoeff])
	for i in range(MaxCoeff):
		amps = np.zeros(MaxCoeff)
		amps[i] = 1
		s = numpy.polynomial.hermite.hermval(x, amps)*np.exp(-0.5*(x)**2)
		s/=np.std(s)
		hermiteMatrix[:,i] = s

	MTM = np.dot(hermiteMatrix.T, hermiteMatrix)

	Prior = 1000.0
	diag=MTM.diagonal().copy()
	diag += 1.0/Prior**2
	np.fill_diagonal(MTM, diag)

	Md = np.dot(hermiteMatrix.T, TScrunched)
	Chol_MTM = sp.linalg.cho_factor(MTM)
	ML = sp.linalg.cho_solve(Chol_MTM, Md)

	s = np.dot(hermiteMatrix, ML)

	r = TScrunched - s	


	loglike  = -0.5*np.sum(r**2)/TSNoise**2

	if(doplot == True):

	    plt.plot(np.linspace(0,1,Nbins[i]), TScrunched)
	    plt.plot(np.linspace(0,1,Nbins[i]),s)
	    plt.show()
	    plt.plot(np.linspace(0,1,Nbins[i]),TScrunched-s)
	    plt.show()

	    zml = ML[0]
	    return ML/zml

	return loglike




parameters = []
parameters.append('Phase')
parameters.append('Width')


print parameters
n_params = len(parameters)
print n_params

    
pmin = np.array(np.ones(n_params))*-1024
pmax = np.array(np.ones(n_params))*1024


x0 = np.array(np.zeros(n_params))

pcount = 0
x0[pcount] = 0.5
pcount += 1

x0[pcount] = 0.1
pcount += 1



cov_diag = np.array(np.ones(n_params))

pcount = 0
cov_diag[pcount] = 0.1
pcount += 1
cov_diag[pcount] = 0.1
pcount += 1


burnin=1000
sampler = ptmcmc.PTSampler(ndim=n_params,logl=InitialLogLike,logp=my_prior,
                            cov=np.diag(cov_diag**2),
                            outDir='./Initchains/',
                            resume=True)

sampler.sample(p0=x0,Niter=10000,isave=10,burn=burnin,thin=1,neff=1000)

chains=np.loadtxt('./Initchains/chain_1.txt').T
ML=chains.T[np.argmax(chains[-3][burnin:])][:n_params]
doplot=True
MLC = InitialLogLike(ML)


