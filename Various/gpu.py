import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import psrchive
from libstempo.libstempo import *
import libstempo as T


import numpy
import theano
import theano.tensor as tt
from theano import pp




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
psr = T.tempopulsar(parfile="OneChan.par", timfile = "OneChan.tim")
psr.fit()
SatSecs = psr.satSec()
SatDays = psr.satDay()
FNames = psr.fnames()
NToAs = psr.nobs


#Check how many timing model parameters we are fitting for (in addition to phase)
numTime=len(psr.pars())
redChisq = psr.chisq()/(psr.nobs-len(psr.pars())-1)
TempoPriors=np.zeros([numTime,2]).astype(np.float64)
for i in range(numTime):
        TempoPriors[i][0]=psr[psr.pars()[i]].val
        TempoPriors[i][1]=psr[psr.pars()[i]].err/np.sqrt(redChisq)
	print "fitting for: ", psr.pars()[i], TempoPriors[i][0], TempoPriors[i][1]


designMatrix=psr.designmatrix(incoffset=False)
for i in range(numTime):
	designMatrix[:,i] *= TempoPriors[i][1]

designMatrix=np.float64(designMatrix)

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

toas=psr.toas()
residuals = psr.residuals(removemean=False)
BatCorrs = psr.batCorrs()
ModelBats = psr.satSec() + BatCorrs - residuals/SECDAY

ProfileStartBats = ProfileInfo[:,2]/SECDAY + ProfileInfo[:,3]*0 + ProfileInfo[:,3]*0.5 + BatCorrs
ProfileEndBats =  ProfileInfo[:,2]/SECDAY + ProfileInfo[:,3]*(ProfileInfo[:,4]-1) + ProfileInfo[:,3]*0.5 + BatCorrs

Nbins = ProfileInfo[:,4]
ProfileBinTimes = []
for i in range(NToAs):
	ProfileBinTimes.append((np.linspace(ProfileStartBats[i], ProfileEndBats[i], Nbins[i])-ModelBats[i])*SECDAY)
ShiftedBinTimes = np.float64(np.array(ProfileBinTimes))

FlatBinTimes = (ShiftedBinTimes.flatten())[:useToAs*1024]

ReferencePeriod = np.float64(ProfileInfo[0][5])

#MaxCoeff = 10
#InterpBasis, InterpolatedTime = PreComputeShapelets(interpTime = 1, MeanBeta = 47.9)



#from pymc3 import Model, Normal, HalfNormal, Uniform, theano
#from theano import tensor as tt
#from theano import function





def uniformTransform(x, pmin, pmax):
	return np.log((x - pmin) / (pmax - x))

def uniformErrTransform(x, err, pmin, pmax):

	t1 = uniformTransform(x, pmin, pmax)
	t2 = uniformTransform(x+err, pmin, pmax)
	t3 = uniformTransform(x-err, pmin, pmax)

	diff1=t2-t1
	diff2=t1-t3

	return 0.5*(diff1+diff2)




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

		#smean = np.sum(s)/Nbins 
		#s = s-smean

		#sstd = np.dot(s,s)/Nbins
		#s=s/np.sqrt(sstd)

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

	d = {'amplitude': amplitude, 'noise_log_': noise_log_, 'offset': offset, 'phase_interval_': uniformTransform(0.00288206, 0, ReferencePeriod)}

	return d




def hessian(useToAs):
	pnoise=ProfileInfo[:useToAs,6]**2
	onehess=1.0/np.float64(ProfileInfo[:useToAs,4]/pnoise)
	noisehess = 1.0/(ProfileInfo[:useToAs,4]*(3.0/(ProfileInfo[:useToAs,6]*ProfileInfo[:useToAs,6]) - 1.0/(ProfileInfo[:useToAs,6]*ProfileInfo[:useToAs,6])))
	d = {'amplitude': np.float64(onehess), 'noise_log_': np.float64(noisehess), 'offset': np.float64(onehess), 'phase_interval_': np.ones(1)*uniformErrTransform(0.00288206, 1.0559855345705067e-07,  0, ReferencePeriod)**2}

	return d


MLpoint = ML(useToAs)
hess = hessian(useToAs)

start = MLpoint


from theano import function, config, shared, sandbox
import theano.sandbox.cuda.basic_ops
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

theano.config.compute_test_value='off'


#parameters that define a two gaussian model
gsep    = np.float32(Savex[1]*ReferencePeriod/1024)
g1width = np.float32(Savex[2]*ReferencePeriod/1024)
g2width = np.float32(Savex[3]*ReferencePeriod/1024)
g2amp   = np.float32(Savex[4])


Tg1width=theano.shared(g1width)
Tg2width =theano.shared(g2width)
Tg2amp=theano.shared(g2amp)
Tgsep=theano.shared(gsep)

TRP = theano.shared(np.float32(ReferencePeriod))
TBinTimes = theano.shared(np.float32(ShiftedBinTimes))


TFlatTimes = theano.shared(np.float32(FlatBinTimes))
FlatData = (ProfileData.flatten())[:useToAs*1024]
TFlatData = theano.shared(np.float32(FlatData))

amps   = tt.vector('amps', dtype='float32')
offs   = tt.vector('offs', dtype='float32')
sigs   = tt.vector('sigs', dtype='float32')
phase  = tt.scalar('phase', dtype='float32')

x = ( TFlatTimes - phase + TRP/2) % (TRP ) - TRP/2
y = tt.exp(-0.5*(x)**2/Tg1width**2)

MakeXVec = theano.function([phase], x)
MakeYVec = theano.function([x], y)

theano_rng = RandomStreams(189)


pval = theano.function([], theano_rng.normal( size = (1,), avg = 0.0, std = 1.0, dtype=theano.config.floatX))


start = time.clock()


for i in range(20000):
	if(i%1000 == 0):
		print i



	xvec = MakeXVec(pval()[0])
	yvec = MakeYVec(xvec)


end = time.clock()

print "time", start-end




MakeXVecGPU = theano.function([phase],sandbox.cuda.basic_ops.gpu_from_host(x))
MakeYVecGPU = theano.function([x], sandbox.cuda.basic_ops.gpu_from_host(y))


theano_rng = RandomStreams(189)


pval = theano.function([], theano_rng.normal( size = (1,), avg = 0.0, std = 1.0, dtype=theano.config.floatX))



start = time.clock()


for i in range(20000):
	if(i%1000 == 0):
		print i



	#x = MakeXVec(pval()[0])
	x = MakeXVecGPU(pval()[0])


end = time.clock()

print "time", start-end


