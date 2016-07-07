import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import psrchive
from libstempo.libstempo import *
import libstempo as T




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

SECDAY = 24*60*60

#First load pulsar.  We need the sats (separate day/second), and the file names of the archives (FNames)
psr = T.tempopulsar(parfile="OneProf.par", timfile = "OneChan.tim")
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
        
        #print "Subint Info:", i, nbins, nchans, npols, foldingperiod, inttime, centerfreq
        
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
                    ProfileInfo.append([SatSecs[profcount], SatDays[profcount], np.float128(intsec)+np.float128(fracsecs), pulsesamplerate, nbins, foldingperiod, noiselevel])                    
                    #print "ChanInfo:", j, chanfreq, toafreq
                    profcount += 1
                    if(profcount == NToAs):
                        break

len(ProfileData)
ProfileInfo=np.array(ProfileInfo)
ProfileData = np.array(ProfileData)

parameters = []
for i in range(NToAs):
	parameters.append('Amp'+str(i))
for i in range(NToAs):
	parameters.append('BL'+str(i))
for i in range(NToAs):
	parameters.append('Noise'+str(i))

n_params = len(parameters)
print n_params


Savex = np.array(np.zeros(5))

Savex[0] = -6.30581674e-01
Savex[1] = 9.64886554e+01
Savex[2] = 3.12774568e+01
Savex[3] = 2.87192467e+01
Savex[4] = 1.74380328e+00


useToAs=100


from pymc3 import Model, Normal, HalfNormal

basic_model = Model()

with basic_model:

	# Priors for unknown model parameters
	amplitude = Normal('amplitude', mu=0, sd=10000, shape = useToAs)
	offset = Normal('offset', mu=0, sd=10000, shape = useToAs)
	noise = HalfNormal('noise', sd=10000, shape = useToAs)

	ReferencePeriod = ProfileInfo[0][5]
	FoldingPeriodDays = ReferencePeriod/SECDAY

	phase   = Savex[0]*ReferencePeriod/SECDAY
	gsep    = Savex[1]*ReferencePeriod/SECDAY/1024
	g1width = np.float64(Savex[2]*ReferencePeriod/SECDAY/1024)
	g2width = np.float64(Savex[3]*ReferencePeriod/SECDAY/1024)
	g2amp   = Savex[4]

	toas=psr.toas()
	residuals = psr.residuals(removemean=False)
	BatCorrs = psr.batCorrs()
	ModelBats = psr.satSec() + BatCorrs - phase - residuals/SECDAY

	loglike = 0
	Y_obs = 0
	for i in range(useToAs):

		'''Start by working out position in phase of the model arrival time'''

		ProfileStartBat = ProfileInfo[i,2]/SECDAY + ProfileInfo[i,3]*0 + ProfileInfo[i,3]*0.5 + BatCorrs[i]
		ProfileEndBat = ProfileInfo[i,2]/SECDAY + ProfileInfo[i,3]*(ProfileInfo[i,4]-1) + ProfileInfo[i,3]*0.5 + BatCorrs[i]


		Nbins = ProfileInfo[i,4]
		x=np.linspace(ProfileStartBat, ProfileEndBat, Nbins)

		minpos = ModelBats[i] - FoldingPeriodDays/2
		if(minpos < ProfileStartBat):
			minpos=ProfileStartBat

		maxpos = ModelBats[i] + FoldingPeriodDays/2
		if(maxpos > ProfileEndBat):
			maxpos = ProfileEndBat


		'''Need to wrap phase for each of the Gaussian components separately'''

		BinTimes = x-ModelBats[i]
		BinTimes[BinTimes > maxpos-ModelBats[i]] = BinTimes[BinTimes > maxpos-ModelBats[i]] - FoldingPeriodDays
		BinTimes[BinTimes < minpos-ModelBats[i]] = BinTimes[BinTimes < minpos-ModelBats[i]] + FoldingPeriodDays

		BinTimes=np.float64(BinTimes)

		s = 1.0*np.exp(-0.5*(BinTimes)**2/g1width**2)


		BinTimes = x-ModelBats[i]-gsep
		BinTimes[BinTimes > maxpos-ModelBats[i]-gsep] = BinTimes[BinTimes > maxpos-ModelBats[i]-gsep] - FoldingPeriodDays
		BinTimes[BinTimes < minpos-ModelBats[i]-gsep] = BinTimes[BinTimes < minpos-ModelBats[i]-gsep] + FoldingPeriodDays

		BinTimes=np.float64(BinTimes)

		s += g2amp*np.exp(-0.5*(BinTimes)**2/g2width**2)

		'''Now subtract mean and scale so std is one.'''

		smean = np.sum(s)/Nbins 
		s = s-smean

		sstd = np.dot(s,s)/Nbins
		s=s/np.sqrt(sstd)


		# Expected value of outcome
		signal = s*amplitude[i] + offset[i]


		# Likelihood (sampling distribution) of observations
		Y_obs += Normal('Y_obs', mu=signal, sd=noise[i], observed=ProfileData[i])



'''
from pymc3 import find_MAP

map_estimate = find_MAP(model=basic_model)

print(map_estimate)

from scipy import optimize

map_estimate = find_MAP(model=basic_model, fmin=optimize.fmin_powell)

print(map_estimate)
'''


def ML(useToAs):


	ReferencePeriod = ProfileInfo[0][5]
	FoldingPeriodDays = ReferencePeriod/SECDAY
	phase=Savex[0]*ReferencePeriod/SECDAY

	pcount = numTime+1

	phase   = Savex[0]*ReferencePeriod/SECDAY
	gsep    = Savex[1]*ReferencePeriod/SECDAY/1024
	g1width = np.float64(Savex[2]*ReferencePeriod/SECDAY/1024)
	g2width = np.float64(Savex[3]*ReferencePeriod/SECDAY/1024)
	g2amp   = Savex[4]


	toas=psr.toas()
	residuals = psr.residuals(removemean=False)
	BatCorrs = psr.batCorrs()
	ModelBats = psr.satSec() + BatCorrs - phase - residuals/SECDAY


	amplitude=np.zeros(useToAs)
	noise_log_=np.zeros(useToAs)
	offset = np.zeros(useToAs)
	for i in range(useToAs):

		'''Start by working out position in phase of the model arrival time'''

		ProfileStartBat = ProfileInfo[i,2]/SECDAY + ProfileInfo[i,3]*0 + ProfileInfo[i,3]*0.5 + BatCorrs[i]
		ProfileEndBat = ProfileInfo[i,2]/SECDAY + ProfileInfo[i,3]*(ProfileInfo[i,4]-1) + ProfileInfo[i,3]*0.5 + BatCorrs[i]

		Nbins = ProfileInfo[i,4]
		x=np.linspace(ProfileStartBat, ProfileEndBat, Nbins)

		minpos = ModelBats[i] - FoldingPeriodDays/2
		if(minpos < ProfileStartBat):
			minpos=ProfileStartBat

		maxpos = ModelBats[i] + FoldingPeriodDays/2
		if(maxpos > ProfileEndBat):
			maxpos = ProfileEndBat


		'''Need to wrap phase for each of the Gaussian components separately.  Fortran style code incoming'''

		BinTimes = x-ModelBats[i]
		BinTimes[BinTimes > maxpos-ModelBats[i]] = BinTimes[BinTimes > maxpos-ModelBats[i]] - FoldingPeriodDays
		BinTimes[BinTimes < minpos-ModelBats[i]] = BinTimes[BinTimes < minpos-ModelBats[i]] + FoldingPeriodDays

		BinTimes=np.float64(BinTimes)
		    
		s = 1.0*np.exp(-0.5*(BinTimes)**2/g1width**2)


		BinTimes = x-ModelBats[i]-gsep
		BinTimes[BinTimes > maxpos-ModelBats[i]-gsep] = BinTimes[BinTimes > maxpos-ModelBats[i]-gsep] - FoldingPeriodDays
		BinTimes[BinTimes < minpos-ModelBats[i]-gsep] = BinTimes[BinTimes < minpos-ModelBats[i]-gsep] + FoldingPeriodDays

		BinTimes=np.float64(BinTimes)

		s += g2amp*np.exp(-0.5*(BinTimes)**2/g2width**2)

		'''Now subtract mean and scale so std is one.  Makes the matrix stuff stable.'''

		smean = np.sum(s)/Nbins 
		s = s-smean

		sstd = np.dot(s,s)/Nbins
		s=s/np.sqrt(sstd)

		'''Make design matrix.  Two components: baseline and profile shape.'''

		M=np.ones([2,Nbins])
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

		baseline=dNMMNM[0]
		amp = dNMMNM[1]
		noise = np.std(ProfileData[i] - baseline - amp*s)

		amplitude[i] = amp
		noise_log_[i] = np.log(noise)
		offset[i] = baseline

		#print "ML", amp, baseline, noise

	d = {'amplitude': amplitude, 'noise_log_': noise_log_, 'offset': offset}

	return d


def hessian(useToAs):
	pnoise=ProfileInfo[:useToAs,6]**2
	onehess=1.0/np.float64(ProfileInfo[:useToAs,4]/pnoise)
	noisehess = 1.0/(ProfileInfo[:useToAs,4]*(3.0/(ProfileInfo[:useToAs,6]*ProfileInfo[:useToAs,6]) - 1.0/(ProfileInfo[:useToAs,6]*ProfileInfo[:useToAs,6])))
	d = {'amplitude': np.float64(onehess), 'noise_log_': np.float64(noisehess), 'offset': np.float64(onehess)}

	return d



MLpoint = ML(useToAs)
hess = hessian(useToAs)

from pymc3 import HamiltonianMC, sample

with basic_model:

	# Use starting ML point
	start = MLpoint

	hess = hessian(useToAs)

	#Set scaling using hessian
	step = HamiltonianMC(scaling=basic_model.dict_to_array(hess), is_cov=True)
	# draw 2000 posterior samples
	trace = sample(2000, start=start)

from pymc3 import NUTS, sample

with basic_model:

	# Use starting ML point
	start = MLpoint

	hess = hessian(useToAs)

	#Set scaling using hessian
	step = NUTS(scaling=basic_model.dict_to_array(hess), is_cov=True)
	# draw 2000 posterior samples
	trace = sample(2000, start=start)

from pymc3 import traceplot

traceplot(trace);


