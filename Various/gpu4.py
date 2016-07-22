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
                        hermiteMatrix[:,i] = s/np.std(s)
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

def TNothpl(n,x, pl):


	a=2.0
	c=1.0
	y0=1.0
	y1=2.0*x



	pl=tt.set_subtensor(pl[0], tt.ones_like(TFlatBinTimes))
	pl=tt.set_subtensor(pl[1], 2*x)
	
	for k in range(2, n):

		c=2.0*(k-1.0)

		y0=y0/tt.sqrt((k*1.0))
		y1=y1/tt.sqrt((k*1.0))
		#yn=(a*x+b)*y1-c*y0

		yn = 2*x*y1 - c*y0


		pl=tt.set_subtensor(pl[k], yn)
		y0=y1
		y1=yn
	
	return pl

  
def Bconst(width, i):
	return (1.0/np.sqrt(width))/np.sqrt(2.0**i*np.sqrt(np.pi))



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


Tg1width=theano.shared(g1width, name='g1width')
Tg2width =theano.shared(g2width, name='g2width')
Tg2amp=theano.shared(g2amp, name='g2amp')
Tgsep=theano.shared(gsep, name='gsep')

TRP = theano.shared(np.float32(ReferencePeriod), name='TRP')
TBinTimes = theano.shared(np.float32(ShiftedBinTimes), name='TBinTimes')
FlatBinTimes = np.float32(ShiftedBinTimes).flatten()
TFlatBinTimes = theano.shared(np.float32(ShiftedBinTimes).flatten(), name='TBinTimes')
TData = theano.shared(np.float32(ProfileData), name='TData')

Nbins=Nbins.astype(int)
TNbins=theano.shared(Nbins)


AP = np.zeros([useToAs,1])
AP[:,0] = MLpoint.get('amplitude')

OP = np.zeros([useToAs,1])
OP[:,0] = MLpoint.get('offset')

SP = np.zeros([useToAs,1])
SP[:,0] = np.exp(MLpoint.get('noise_log_'))

AmpParams = theano.shared(AP.astype(np.float32), name='AmpParams', broadcastable = [False,True])
OffParams = theano.shared(OP.astype(np.float32), name='OffParams', broadcastable = [False,True])
SigParams = theano.shared(SP.astype(np.float32), name='SigParams', broadcastable = [False,True])


MaxCoeff = 100
BasisVectors = theano.shared(np.float32(np.zeros([MaxCoeff, len(FlatBinTimes)])), name='BasisVectors')

tp = tt.scalar('tp', dtype='float32')

sv = tt.matrix('sv', dtype='float32')
TempPhase=np.float32(0.00288206)

FlatX = ( TFlatBinTimes - tp + TRP/2) % (TRP ) - TRP/2

SMatrix = TNothpl(MaxCoeff,FlatX, BasisVectors)*tt.exp(-0.5*TFlatBinTimes)


ShapeCoeff = theano.shared(np.float32(np.ones(MaxCoeff)), name='ShapeCoeff')
SVec = (tt.dot(SMatrix.T,ShapeCoeff)).reshape([useToAs, 1024])

lfunc = -0.5*tt.sum(((AmpParams*SVec+OffParams-TData)/SigParams)**2-1)

ReturnLFunc = theano.function(inputs=[tp],outputs= lfunc, name='ReturnLFunc')


getSMatrix = theano.function(inputs=[tp], outputs = SMatrix, name='getSMatrix')
getSVec = theano.function(inputs=[tp], outputs = SVec, name='getSVec')

TNothpl(10,FlatX, sv)

x2 = ( TBinTimes - tp + TRP/2) % (TRP ) - TRP/2

MeanBeta = 0.01*0.0045697659368490011
x = np.linspace(0, 0+4.4626620477041026292e-06*(1024-1), 1024)
x = ( x + 0.0045697659368490011/2) % (0.0045697659368490011) - 0.0045697659368490011/2
x=x/MeanBeta
ZeroExVec = np.exp(-0.5*(x)**2)

TZeroExVec = theano.shared(np.float32(ZeroExVec))



SmallCOSSINMatrix = theano.shared(np.float32(np.ones([1,513,2])), name='SmallCOSSINMatrix')

fftfreqs=theano.shared(np.float32(np.abs(np.fft.fftfreq(1024)[:513])), name='fftfreqs')




def setSmallCovMatrix2():
	
	CosM = tt.set_subtensor(SmallCOSSINMatrix[:,:,0], Rtest)
	CosM = tt.set_subtensor(CosM[:,:,1], ITest)

	return CosM

IRFFT = tt.fft.irfft(setSmallCovMatrix2())


RFFT = tt.fft.rfft(TZeroExVec.dimshuffle('x', 0))
Rtest=RFFT[0][:,0]*tt.cos(2*np.pi*231*fftfreqs) + RFFT[0][:,1]*tt.sin(2*np.pi*231*fftfreqs)
ITest=RFFT[0][:,1]*tt.cos(2*np.pi*231*fftfreqs) + RFFT[0][:,0]*tt.sin(2*np.pi*231*fftfreqs)
RVec = tt.fft.rfft(IVec)*setCovMatrix() 



ERFFT = RFFT*setSmallCovMatrix()
getERFFT=theano.function(inputs=[]


Shifted=np.fft.irfft(np.fft.rfft(np.float64(ZeroExVec))*np.exp(1*2*np.pi*231*rfftfreqs*1j))
RShifted=np.real(Shifted)/np.max(np.real(Shifted))




zeros=theano.shared((np.zeros(100).astype(int)), name='zeros')

MaxCoeff = 1
InterpBasis, InterpolatedTime = PreComputeShapelets(interpTime = 1, MeanBeta = 47.9)

TInterpTime = theano.shared(np.float32(InterpolatedTime), name='TInterpTime')

tp = tt.scalar('tp', dtype='float32')
TempPhase=np.float32(0.00288206)
Subx2 = ( TBinTimes[:,0] - tp + TRP/2) % (TRP ) - TRP/2


IBin = tt.cast(Subx2%(TRP/TNbins)/TInterpTime, dtype='int32')
getIBin = theano.function(inputs=[tp], outputs = IBin, name='getIBin')


WBT = tt.cast(tt.round((TBinTimes[:,0]-TInterpTime*IBin)/(TRP/TNbins)), dtype='int32')
getWBT = theano.function(inputs=[tp], outputs = WBT, name='getWBT')


TestCoeffs=np.zeros(MaxCoeff)
TestCoeffs[0] = 1
ShapeCoeff = theano.shared(np.float32(TestCoeffs), name='ShapeCoeff')
TInterpBasis = theano.shared(np.float32(InterpBasis))

IVec = tt.dot(TInterpBasis[IBin],ShapeCoeff)
getIVec = theano.function(inputs=[tp], outputs = IVec, name='getIVec')


def MyRVec(iv):
	
	for i in range(useToAs):
		temp=tt.zeros_like(iv[i])
		rollVal=-WBT[i]
		temp=tt.set_subtensor(temp[rollVal:], iv[i][:-rollVal])
		temp=tt.set_subtensor(temp[:rollVal], iv[i][-rollVal:])
		iv = tt.set_subtensor(iv[i], temp)

	return iv

lfunc = -0.5*tt.sum(((AmpParams*MyRVec(IVec)+OffParams-TData)/SigParams)**2-1)
ReturnLFunc = theano.function(inputs=[tp],outputs= lfunc, name='ReturnLFunc')





getIBin = theano.function(inputs=[], outputs = zeros, name='getIBin')


WBT = tt.cast(tt.round((TBinTimes[:,0]-TInterpTime*zeros)/(TRP/TNbins)), dtype='int32')
getWBT = theano.function(inputs=[], outputs = WBT, name='getWBT')


TestCoeffs=np.zeros(MaxCoeff)
TestCoeffs[0] = 1
ShapeCoeff = theano.shared(np.float32(TestCoeffs), name='ShapeCoeff')
TInterpBasis = theano.shared(np.float32(InterpBasis))

IVec = tt.dot(TInterpBasis[zeros],ShapeCoeff)
getIVec = theano.function(inputs=[], outputs = IVec, name='getIVec')











COSSINMatrix = theano.shared(np.float32(np.ones([100,513,2])), name='sincosMatrix')

fftfreqs=theano.shared(np.float32(np.abs(np.fft.fftfreq(1024)[:513])), name='fftfreqs')

def setCovMatrix():
	
	CosM = tt.set_subtensor(COSSINMatrix[:,:,0], tt.cos(2*np.pi*tt.outer(WBT,fftfreqs)))
	CosM = tt.set_subtensor(CosM[:,:,1], tt.sin(2*np.pi*tt.outer(WBT,fftfreqs)))

	return CosM

RVec = tt.fft.rfft(IVec)*setCovMatrix() #tt.fft.irfft(tt.fft.rfft(IVec)*setCovMatrix())

CM = setCovMatrix()
getRVec = theano.function(inputs=[tp], outputs = RVec, name='getRVec')

getCM = theano.function(inputs=[tp], outputs = CM, name='getCM')

#getEVec = theano.function(inputs=[tp], outputs = EMatrix, name='getEVec')


def RVec(iv):
	for i in range(useToAs):
		iv=tt.set_subtensor(iv[i], tt.roll(iv[i], -WBT[i]))
	return iv






lfunc = -0.5*tt.sum(((AmpParams*RVec(IVec)+OffParams-TData)/SigParams)**2-1)
#gfunc  = tt.grad(lfunc, tp)

#ReturnAll = theano.function(inputs=[tp],outputs= [lfunc, gfunc], name='ReturnAll')
ReturnLFunc = theano.function(inputs=[tp],outputs= lfunc, name='ReturnLFunc')


lsum=0
start = time.clock()


for i in range(1000):
	if(i%1000 == 0):
		print i

	TempPhase = np.random.normal(0,1)#
	#ib = getIVec(np.float32(TempPhase))
	#l,g = ReturnAll(np.float32(TempPhase))
	#l = ReturnLFunc(np.float32(TempPhase))
	#getSVec(TempPhase)

	RFFT = tt.fft.rfft(TZeroExVec.dimshuffle('x', 0))
	Rtest=RFFT[0][:,0]*tt.cos(2*np.pi*231*fftfreqs) + RFFT[0][:,1]*tt.sin(2*np.pi*231*fftfreqs)
	ITest=RFFT[0][:,1]*tt.cos(2*np.pi*231*fftfreqs) + RFFT[0][:,0]*tt.sin(2*np.pi*231*fftfreqs)
	RVec = tt.fft.rfft(IVec)*setCovMatrix() 
	
	lsum += TempPhase

end = time.clock()

print "time", end - start

