
def MarginLogLike(x):
    


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


	#Multiply and shift out the shapelet model

	#ShapeCoeff = np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*lfunc.MLShapeCoeff, axis=1)

	np.dot(lfunc.InterpBasis[InterpBins[i]][:,:NCoeff+1], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1))

	s=[np.roll(np.dot(lfunc.InterpBasis[InterpBins[i]][:,:NCoeff+1], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]

	#j=[np.roll(np.dot(lfunc.InterpJitterMatrix[InterpBins[i]][:,:NCoeff+1], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*ShapeAmps, axis=1)), -RollBins[i]) for i in range(len(RollBins))]



	#Subtract mean and rescale


	s = [s[i] - np.sum(s[i])/lfunc.Nbins[i] for i in range(lfunc.NToAs)]	
	s = [s[i]/(np.dot(s[i],s[i])/lfunc.Nbins[i]) for i in range(lfunc.NToAs)]


	for i in range(lfunc.NToAs):


		'''Make design matrix.  Two components: baseline and profile shape.'''

		M=np.ones([2,lfunc.Nbins[i]])
		M[1] = s[i]


		pnoise = lfunc.ProfileInfo[i][6]

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
		    
		    
		dNM = np.dot(lfunc.ProfileData[i], M.T)/(pnoise*pnoise)


		dNMMNM = np.dot(dNM.T, InvMNM)
		MarginLike = np.dot(dNMMNM, dNM)

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
