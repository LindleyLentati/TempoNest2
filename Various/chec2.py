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



theano.config.compute_test_value='off'


#parameters that define a two gaussian model
gsep    = np.float64(Savex[1]*ReferencePeriod/1024)
g1width = np.float64(Savex[2]*ReferencePeriod/1024)
g2width = np.float64(Savex[3]*ReferencePeriod/1024)
g2amp   = np.float64(Savex[4])


Tg1width=theano.shared(g1width)
Tg2width =theano.shared(g2width)
Tg2amp=theano.shared(g2amp)
Tgsep=theano.shared(gsep)

TRP = theano.shared(np.float64(ReferencePeriod))
TBinTimes = theano.shared(np.float64(ShiftedBinTimes))


TFlatTimes = theano.shared(np.float64(FlatBinTimes))
FlatData = (ProfileData.flatten())[:useToAs*1024]
TFlatData = theano.shared(np.float64(FlatData))

amps   = tt.vector('amps', dtype='float64')
offs   = tt.vector('offs', dtype='float64')
sigs   = tt.vector('sigs', dtype='float64')
phase  = tt.scalar('phase', dtype='float64')

x = ( TFlatTimes - phase + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2
y = tt.exp(-0.5*(x)**2/Tg1width**2)
x2 = ( TFlatTimes - phase - gsep + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2
y2 = g2amp*tt.exp(-0.5*(x2)**2/Tg2width**2)


AmpVec = theano.tensor.extra_ops.repeat(amps, 1024)
OffVec = theano.tensor.extra_ops.repeat(offs, 1024)
SigVec = theano.tensor.extra_ops.repeat(sigs, 1024)

Nbins=Nbins.astype(int)
TNbins=theano.shared(Nbins)

s = AmpVec*(y+y2) + OffVec
'''
like = 0.5*tt.sum(((TFlatData-s)/SigVec)**2)  + 0.5*tt.sum(TNbins[:useToAs]*tt.log(sigs**2))

glike = tt.grad(like, [phase, amps, offs, sigs])

getS = theano.function([phase, amps, offs], s)
getX = theano.function([phase, amps, offs, sigs], like)	
getG = theano.function([phase, amps, offs, sigs], glike)


def TheanoFunc2(theta):

	phaseval=theta[0]
	ampvec=theta[1:1+useToAs]
	offvec=theta[1+useToAs:1+2*useToAs]
	sigvec=theta[1+2*useToAs:1+3*useToAs]
	
	return getX(phaseval, ampvec, offvec, sigvec)*1, np.array(getG(phaseval, ampvec, offvec, sigvec))#, getS(phaseval, ampvec, offvec)



import nuts
from nuts import nuts6


M = 5000
Madapt = 5000
theta0 = np.array([0.00288206, start.get('amplitude'), start.get('offset'), np.exp(start.get('noise_log_'))])
epsilon = np.sqrt(np.array([1.0559855345705067e-14, hess.get('amplitude'), hess.get('offset'), 0.02]))

samples, lnprob, epsilon = nuts6(TheanoFunc2, M, Madapt, theta0, delta=0.2, epsilon = epsilon)

'''
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc

like = -0.5*tt.sum(((TFlatData-s)/SigVec)**2)  - 0.5*tt.sum(TNbins[:useToAs]*tt.log(sigs**2))

glike = tt.grad(like, [phase, amps, offs, sigs])

getS = theano.function([phase, amps, offs], s)
getX = theano.function([phase, amps, offs, sigs], like)	
getG = theano.function([phase, amps, offs, sigs], glike)



def TheanoFuncL(theta):

	phaseval=theta[0]
	ampvec=theta[1:1+useToAs]
	offvec=theta[1+useToAs:1+2*useToAs]
	sigvec=theta[1+2*useToAs:1+3*useToAs]
	
	return getX(phaseval, ampvec, offvec, sigvec)*1


def TheanoFuncG(theta):

	phaseval=theta[0]
	ampvec=theta[1:1+useToAs]
	offvec=theta[1+useToAs:1+2*useToAs]
	sigvec=theta[1+2*useToAs:1+3*useToAs]
        
        g=getG(phaseval, ampvec, offvec, sigvec)
	for i in range(3): g[0] = np.append(g[0], g[1+i])
        
	return getX(phaseval, ampvec, offvec, sigvec)*1, g[0]

def TheanoFuncS(theta):

	phaseval=theta[0]
	ampvec=theta[1:1+useToAs]
	offvec=theta[1+useToAs:1+2*useToAs]
	sigvec=theta[1+2*useToAs:1+3*useToAs]
	
	return getS(phaseval, ampvec, offvec)*1

parameters = []
parameters.append('phase')
for i in range(useToAs):
	parameters.append('Amp'+str(i))
for i in range(useToAs):
	parameters.append('BL'+str(i))
for i in range(useToAs):
	parameters.append('Noise'+str(i))

n_params = len(parameters)
print n_params

theta0 = np.array([0.00288206])
theta0 = np.append(theta0, start.get('amplitude'))
theta0 = np.append(theta0, start.get('offset'))
theta0 = np.append(theta0, np.exp(start.get('noise_log_')))


epsilon = np.array([1.0559855345705067e-14])
epsilon = np.append(epsilon, hess.get('amplitude'))
epsilon = np.append(epsilon, hess.get('offset'))
epsilon = np.append(epsilon, np.ones(useToAs)*0.005)


cov=np.diag(epsilon)

pmin = np.ones(n_params)*-1000
pmax = np.ones(n_params)*1000

    
def lnpriorfn(x):

	if np.all(pmin < x) and np.all(pmax > x):
	    return 0.0
	else:
	    return -np.inf  
	return 0.0

def lnpriorfn_grad(x):
	return lnpriorfn(x), np.zeros_like(x)

doplot=False
burnin=1000
sampler = ptmcmc.PTSampler(n_params, TheanoFuncL, lnpriorfn, np.copy(cov),
                                  logl_grad=TheanoFuncG, logp_grad=lnpriorfn_grad,
                                  outDir='./HMCchains')

sampler.sample(theta0, 10000, burn=burnin, thin=1,
               SCAMweight=10, AMweight=10, DEweight=10, NUTSweight=10, HMCweight=10, MALAweight=0,
               HMCsteps=10, HMCstepsize=0.4)



chains=np.loadtxt('HMCchains/chain_1.txt').T
ML=chains.T[np.argmax(chains[-4])][:n_params]
for i in range(n_params):
    print i, np.std(chains[i])

TestPhase = 0.00288206
#Testphase = 0.0028798#-0.00626009
Testoffs=start.get('offset')
Testamps=start.get('amplitude')

ovec = np.repeat(Testoffs, 1024)
avec = np.repeat(Testamps, 1024)

xvec = FlatBinTimes-TestPhase
xvec = ( xvec + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2

Tests =  np.exp(-0.5*(xvec)**2/g1width**2)

xvec = FlatBinTimes-TestPhase-gsep
xvec = ( xvec + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2

Tests += g2amp*np.exp(-0.5*(xvec)**2/g2width**2)

svec = ovec + avec*Tests


pval2=start.get('phase')
xvec2 = FlatBinTimes-pval2
xvec2 = ( xvec2 + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2
s2 =  np.exp(-0.5*(xvec2)**2/g1width**2)

xvec2 = FlatBinTimes-pval2-gsep
xvec2 = ( xvec + ReferencePeriod/2) % (ReferencePeriod ) - ReferencePeriod/2

s += g2amp*np.exp(-0.5*(xvec)**2/g2width**2)

svec = ovec + avec*s


plt.plot(np.linspace(0,useToAs, 1024*useToAs), FlatData, color='black')
plt.plot(np.linspace(0,useToAs, 1024*useToAs), svec, color='red')
plt.show()


'''

pval = 0.00288206
Tpval = theano.shared(pval)



ltot = 0

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

theano_rng = RandomStreams(189)


avals = theano.function([], theano_rng.normal( size = (useToAs,), avg = 0.0, std = 1.0, dtype=theano.config.floatX))
ovals = theano.function([], theano_rng.normal( size = (useToAs,), avg = 0.0, std = 1.0, dtype=theano.config.floatX))
nvals = theano.function([], theano_rng.normal( size = (useToAs,), avg = 0.0, std = 1.0, dtype=theano.config.floatX)**2)


start = time.clock()


for i in range(20000):
	if(i%100 == 0):
		print i



	l, g = TheanoFunc2(pval, avals(), ovals(), nvals())

	ltot += l

end = time.clock()

print "time", start-end
'''

