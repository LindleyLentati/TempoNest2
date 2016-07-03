import numpy as np
import psrchive
from libstempo.libstempo import *
import libstempo as T
import matplotlib.pyplot as plt
import PTMCMCSampler
from PTMCMCSampler import PTMCMCSampler as ptmcmc

#Fcuntion to determine an estimate of the white noise in the profile data
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
psr = T.tempopulsar(parfile="OneEpoch.par", timfile = "OneEpoch.tim")
psr.fit()
SatSecs = psr.satSec()
SatDays = psr.satDay()
FNames = psr.fnames()
NToAs = psr.nobs


#Check how many timing model parameters we are fitting for (in addition to phase)
numTime=len(psr.pars())
TempoPriors=np.zeros([numTime,2]).astype(np.float128)
for i in range(numTime):
        TempoPriors[i][0]=psr[psr.pars()[i]].val
        TempoPriors[i][1]=psr[psr.pars()[i]].err
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
                    ProfileData.append(profamps)
                    ProfileInfo.append([SatSecs[profcount], SatDays[profcount], np.float128(intsec)+np.float128(fracsecs), pulsesamplerate, nbins, foldingperiod, noiselevel])                    
                    print "ChanInfo:", j, chanfreq, toafreq
                    profcount += 1
                    if(profcount == NToAs):
                        break

len(ProfileData)
ProfileInfo=np.array(ProfileInfo)


def my_prior(x):
    logp = 0.

    if np.all(x <= pmax) and np.all(x >= pmin):
        logp = np.sum(np.log(1/(pmax-pmin)))
    else:
        logp = -np.inf

    return logp

def MarginLogLike(x):
    
    ReferencePeriod = ProfileInfo[0][5]
    FoldingPeriodDays = ReferencePeriod/SECDAY
    phase=x[0]*ReferencePeriod/SECDAY
    TimingParameters=np.float128(x[1:numTime+1])

    pcount = numTime+1

    gsep=x[pcount+0]*ReferencePeriod/SECDAY/1024
    g1width=x[pcount+1]*ReferencePeriod/SECDAY/1024
    g2width=x[pcount+2]*ReferencePeriod/SECDAY/1024
    g2amp=x[pcount+3]

    for i in range(numTime):
        psr[psr.pars()[i]].val = TempoPriors[i][0] + TempoPriors[i][1]*TimingParameters[i]
    
    
    toas=psr.toas()
    residuals = psr.residuals(removemean=False)
    ModelBats = psr.satSec() + psr.batCorrs() - phase - residuals/SECDAY
    
    loglike = 0
    for i in range(NToAs):
    
        '''Start by working out position in phase of the model arrival time'''

        ProfileStartBat = ProfileInfo[i,2]/SECDAY + ProfileInfo[i,3]*0 + ProfileInfo[i,3]*0.5 + psr.batCorrs()[i]
        ProfileEndBat = ProfileInfo[i,2]/SECDAY + ProfileInfo[i,3]*(ProfileInfo[i,4]-1) + ProfileInfo[i,3]*0.5 + psr.batCorrs()[i]
        
        #print ProfileStartBat, ModelBats[0], ProfileEndBat
        
        #print "PDiff:", (ModelBats[0]-ProfileStartBat), (ModelBats[0]-ProfileStartBat)*24*60*60,  (ModelBats[0]-ProfileStartBat)*24*60*60*psr['F0'].val
        
        #print psr.stoas[0], ModelBats[0], psr.satSec()[0], psr.batCorrs()[0], phase, residuals[0]
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
	'''
        for j in range(Nbins):
            if(BinTimes[j] < minpos-ModelBats[i]):
                BinTimes[j] = ProfileEndBat-ModelBats[i]+(j+1)*ProfileInfo[i,3]
            elif(BinTimes[j] > maxpos-ModelBats[i]):
                 BinTimes[j] = ProfileStartBat-ModelBats[i]-(Nbins-j)*ProfileInfo[i,3]
        '''     
         ###   for j in range(Nbins):
            #        print i, j, x[j]-ModelBats[i], ReferencePeriod/SECDAY/2, abs(x[j]-ModelBats[i]) < ReferencePeriod/SECDAY/2, BinTimes[j]
            
        s = 1.0*np.exp(-0.5*(BinTimes)**2/g1width**2)
        
        
        BinTimes = x-ModelBats[i]-gsep
	BinTimes[BinTimes > maxpos-ModelBats[i]-gsep] = BinTimes[BinTimes > maxpos-ModelBats[i]-gsep] - FoldingPeriodDays
        BinTimes[BinTimes < minpos-ModelBats[i]-gsep] = BinTimes[BinTimes < minpos-ModelBats[i]-gsep] + FoldingPeriodDays
	'''
        for j in range(Nbins):
            if(BinTimes[j] < minpos-ModelBats[i]-gsep):
                BinTimes[j] = ProfileEndBat-ModelBats[i]-gsep+(j+1)*ProfileInfo[i,3]
            elif(BinTimes[j] > maxpos-ModelBats[i]-gsep):
                 BinTimes[j] = ProfileStartBat-ModelBats[i]-gsep-(Nbins-j)*ProfileInfo[i,3]
           '''      
                 
        s += g2amp*np.exp(-0.5*(BinTimes)**2/g2width**2)
        
        '''Now subtract mean and scale so std is one.  Makes the matrix stuff stable.'''
      
        
        s=(s-np.mean(s))
        s=s/np.std(s)
        
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
        MarginLike = np.dot(dNMMNM, dNM)
        
        profilelike = -0.5*(logdetMNM - MarginLike)
        loglike += profilelike
        
        if(doplot == True):
            baseline=dNMMNM[0]
            amp = dNMMNM[1]
            plt.plot(x, ProfileData[i])
            plt.plot(x,baseline+s*amp)
            plt.show()
            
    return loglike

parameters = []
parameters.append('Phase')
for i in range(numTime):
	parameters.append(psr.pars()[i])
parameters.append('GSep')
parameters.append('G1Width')
parameters.append('G2Width')
parameters.append('G2Amp')

print parameters
n_params = len(parameters)
print n_params

    
pmin = np.array(np.ones(n_params))*-1024
pmax = np.array(np.ones(n_params))*1024



x0 = np.array(np.zeros(n_params))

x0[0] = -6.30581674e-01
for i in range(numTime):
	x0[1+i] = 0

pcount=numTime+1
x0[pcount+0] = 9.64886554e+01
x0[pcount+1] = 3.12774568e+01
x0[pcount+2] = 2.87192467e+01
x0[pcount+3] = 1.74380328e+00

cov_diag = np.array(np.ones(n_params))
cov_diag[0] = 0.000115669002113
for i in range(numTime):
        cov_diag[1+i] = 1
pcount=numTime+1
cov_diag[pcount+0] = 0.107382829786
cov_diag[pcount+1] = 0.148923616311
cov_diag[pcount+2] = 0.0672450249764
cov_diag[pcount+3] = 0.00524526749377


doplot=False
sampler = ptmcmc.PTSampler(ndim=n_params,logl=MarginLogLike,logp=my_prior,
                            cov=np.diag(cov_diag**2),
                            outDir='./chains/',
                            resume=True)
sampler.sample(p0=x0,Niter=2000,isave=10,burn=1000,thin=1,neff=1000)


chains=np.loadtxt('./chains/chain_1.txt').T
ML=chains.T[np.argmax(chains[5])][:n_params]
doplot=True
MarginLogLike(ML)
