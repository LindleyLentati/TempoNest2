x=x0

SplitBasis = np.zeros([len(lfunc.InterpFBasis), 2*lfunc.NFBasis, lfunc.MaxCoeff])
SplitBasis[:, :lfunc.NFBasis] = np.real(lfunc.InterpFBasis)
SplitBasis[:, lfunc.NFBasis:] = np.imag(lfunc.InterpFBasis)

def FFTMarginLogLike(x):
    


pcount = 0
phase=x[0]*lfunc.ReferencePeriod#lfunc.MeanPhase*lfunc.ReferencePeriod
phasePrior = -0.5*(phase-lfunc.MeanPhase)*(phase-lfunc.MeanPhase)/lfunc.PhasePrior/lfunc.PhasePrior

pcount += 1

NCoeff = lfunc.MaxCoeff-1
#pcount += 1


ShapeAmps=np.zeros([lfunc.MaxCoeff, lfunc.EvoNPoly+1])
ShapeAmps[0][0] = 1
ShapeAmps[1:]=x[pcount:pcount+(lfunc.MaxCoeff-1)*(lfunc.EvoNPoly+1)].reshape([(lfunc.MaxCoeff-1),(lfunc.EvoNPoly+1)])


pcount += (lfunc.MaxCoeff-1)*(lfunc.EvoNPoly+1)

TimingParameters=x[pcount:pcount+lfunc.numTime]
pcount += lfunc.numTime

loglike = 0

TimeSignal = np.dot(lfunc.designMatrix, TimingParameters)

xS = lfunc.ShiftedBinTimes[:,0]-phase

if(lfunc.numTime>0):
	xS -= TimeSignal

xS = ( xS + lfunc.ReferencePeriod/2) % (lfunc.ReferencePeriod ) - lfunc.ReferencePeriod/2

InterpBins = (xS%(lfunc.ReferencePeriod/lfunc.Nbins[:])/lfunc.InterpolatedTime).astype(int)
WBTs = xS-lfunc.InterpolatedTime*InterpBins
RollBins=(np.round(WBTs/(lfunc.ReferencePeriod/lfunc.Nbins[:]))).astype(np.int)

s = [np.dot(ZeroBasis[InterpBins[i]], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)) for i in range(len(RollBins))]
#

Is = [np.dot(lfunc.InterpBasis[InterpBins[i]], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)) for i in range(len(RollBins))]
s = [np.dot(lfunc.InterpFBasis[InterpBins[i]], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)) for i in range(len(RollBins))]
	#s2 = np.sum([np.dot(lfunc.InterpFBasis[InterpBins], ShapeAmps[:,i])*(((lfunc.psr.freqs - lfunc.EvoRefFreq)/1000.0)**i).reshape(lfunc.NToAs,1) for i in range(lfunc.EvoNPoly+1)], axis=0)

Is=[np.roll(Is[i], -RollBins[i]) for i in range(len(RollBins))]
Is = [Is[i] - np.sum(Is[i])/lfunc.Nbins[i] for i in range(lfunc.NToAs)]	



#Rescale



for i in range(lfunc.NToAs):


	rfftfreqs=np.linspace(0,lfunc.NFBasis,lfunc.NFBasis+1)/lfunc.Nbins[i]

	pnoise = lfunc.ProfileInfo[i,6]*np.sqrt(lfunc.Nbins[i])/np.sqrt(2)

	rollVec = np.exp(2*np.pi*RollBins[i]*rfftfreqs*1j)
	rollS1 = s[i]*rollVec
	s[i] = 	rollS1

	#IFFTs = [np.fft.irfft(s[i], n=1024) for i in range(lfunc.NToAs)]	
	FS = np.zeros(2*lfunc.NFBasis)
	FS[:lfunc.NFBasis] = np.real(rollS1[1:])
	FS[lfunc.NFBasis:] = np.imag(rollS1[1:])

	FS /= np.sqrt(np.dot(FS,FS)/(2*lfunc.NFBasis))

	MNM = np.dot(FS, FS)/(pnoise*pnoise)
	detMNM = MNM
	logdetMNM = np.log(detMNM)

	InvMNM = 1.0/MNM

	dNM = np.dot(lfunc.ProfileFData[i], FS)/(pnoise*pnoise)
	dNMMNM = dNM*InvMNM

	MarginLike = dNMMNM*dNM

	profilelike = -0.5*(logdetMNM - MarginLike)
	loglike += profilelike     


		if(lfunc.doplot == True):
		    baseline=dNMMNM[0]
		    amp = dNMMNM[1]
		    noise = np.std(lfunc.ProfileData[i] - baseline - amp*s)
		    print i, amp, baseline, noise
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]), lfunc.ProfileData[i])
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]),baseline+s[i]*amp)
		    plt.show()
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]),lfunc.ProfileData[i]-(baseline+s[i]*amp))
		    plt.show()
		
	return loglike+phasePrior


def FFTMarginLogLike2(x):
    


	pcount = 0
	phase=x[0]*lfunc.ReferencePeriod#lfunc.MeanPhase*lfunc.ReferencePeriod
	phasePrior = -0.5*(phase-lfunc.MeanPhase)*(phase-lfunc.MeanPhase)/lfunc.PhasePrior/lfunc.PhasePrior

	pcount += 1

	NCoeff = lfunc.MaxCoeff-1
	#pcount += 1


	ShapeAmps=np.zeros([lfunc.MaxCoeff, lfunc.EvoNPoly+1])
	ShapeAmps[0][0] = 1
	ShapeAmps[1:]=x[pcount:pcount+(lfunc.MaxCoeff-1)*(lfunc.EvoNPoly+1)].reshape([(lfunc.MaxCoeff-1),(lfunc.EvoNPoly+1)])


	pcount += (lfunc.MaxCoeff-1)*(lfunc.EvoNPoly+1)

	TimingParameters=x[pcount:pcount+lfunc.numTime]
	pcount += lfunc.numTime

	loglike = 0

	TimeSignal = np.dot(lfunc.designMatrix, TimingParameters)

	xS = lfunc.ShiftedBinTimes[:,0]-phase

	if(lfunc.numTime>0):
		xS -= TimeSignal

	xS = ( xS + lfunc.ReferencePeriod/2) % (lfunc.ReferencePeriod ) - lfunc.ReferencePeriod/2

	InterpBins = (xS%(lfunc.ReferencePeriod/lfunc.Nbins[:])/lfunc.InterpolatedTime).astype(int)
	WBTs = xS-lfunc.InterpolatedTime*InterpBins
	RollBins=(np.round(WBTs/(lfunc.ReferencePeriod/lfunc.Nbins[:]))).astype(np.int)


	#return [np.dot(lfunc.InterpFBasis[InterpBins[i]], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)) for i in range(len(RollBins))]
	s = [np.dot(SplitBasis[InterpBins[i]], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)) for i in range(len(RollBins))]


	

	for i in range(lfunc.NToAs):

		rfftfreqs=np.linspace(1,lfunc.NFBasis,lfunc.NFBasis)/lfunc.Nbins[i]

		pnoise = lfunc.ProfileInfo[i,6]*np.sqrt(lfunc.Nbins[i])/np.sqrt(2)

		cosRoll = np.cos(2*np.pi*RollBins[i]*rfftfreqs)
		sinRoll = np.sin(2*np.pi*RollBins[i]*rfftfreqs)

		FS = s[i][:lfunc.NFBasis]*cosRoll + s[i][lfunc.NFBasis:]*sinRoll
		s[i][lfunc.NFBasis:] = s[i][:lfunc.NFBasis]*sinRoll + s[i][lfunc.NFBasis:]*cosRoll
		s[i][:lfunc.NFBasis] = FS

		#rollVec = np.exp(2*np.pi*RollBins[i]*rfftfreqs*1j)
		#rollS1 = s[i]*rollVec
		
		#FS = np.zeros(2*lfunc.NFBasis)
		#FS[:lfunc.NFBasis] = np.real(rollS1)
		#FS[lfunc.NFBasis:] = np.imag(rollS1)

		s[i] /= np.sqrt(np.dot(s[i],s[i])/(2*lfunc.NFBasis))

		MNM = np.dot(s[i], s[i])/(pnoise*pnoise)
		detMNM = MNM
		logdetMNM = np.log(detMNM)

		InvMNM = 1.0/MNM

		dNM = np.dot(lfunc.ProfileFData[i], s[i])/(pnoise*pnoise)
		dNMMNM = dNM*InvMNM

		MarginLike = dNMMNM*dNM

		profilelike = -0.5*(logdetMNM - MarginLike)
		loglike += profilelike     


		if(lfunc.doplot == True):
		    baseline=dNMMNM[0]
		    amp = dNMMNM[1]
		    noise = np.std(lfunc.ProfileData[i] - baseline - amp*s)
		    print i, amp, baseline, noise
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]), lfunc.ProfileData[i])
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]),baseline+s[i]*amp)
		    plt.show()
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]),lfunc.ProfileData[i]-(baseline+s[i]*amp))
		    plt.show()
		
	return loglike+phasePrior


import time
start = time.clock()

ltot=0
for i in range(100):
	if(i%10 == 0):
		print i



	l=FFTMarginLogLike(x)
	ltot+=l


end = time.clock()

print "time", end-start
