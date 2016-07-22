x=np.array([0.2])

#@profile
def PhaseLike(x):
    

	pcount = 0
	phase = x[pcount]*lfunc.ReferencePeriod
	pcount += 1

	loglike = 0

	stepsize=np.zeros([lfunc.MaxCoeff - 1, lfunc.EvoNPoly+1])

	xS = lfunc.ShiftedBinTimes[:,0]-phase

	xS = ( xS + lfunc.ReferencePeriod/2) % (lfunc.ReferencePeriod ) - lfunc.ReferencePeriod/2

	InterpBins = (xS%(lfunc.ReferencePeriod/lfunc.Nbins[:])/lfunc.InterpolatedTime).astype(int)
	WBTs = xS-lfunc.InterpolatedTime*InterpBins
	RollBins=(np.round(WBTs/(lfunc.ReferencePeriod/lfunc.Nbins[:]))).astype(np.int)


	#Multiply and shift out the shapelet model

	
	#s=np.dot(lfunc.InterpBasis[InterpBins], np.sum(lfunc.MLShapeCoeff,axis=1))

	s=[np.roll(np.dot(lfunc.InterpBasis[InterpBins[i]], np.sum(((lfunc.psr.freqs[i] - lfunc.EvoRefFreq)/1000.0)**np.arange(0,lfunc.EvoNPoly+1)*lfunc.MLShapeCoeff, axis=1)), -RollBins[i]) for i in range(len(RollBins))]


	#Subtract mean and rescale


	s = [s[i] - np.sum(s[i])/lfunc.Nbins[i] for i in range(lfunc.NToAs)]	
	s = [s[i]/(np.dot(s[i],s[i])/lfunc.Nbins[i]) for i in range(lfunc.NToAs)]


	for i in range(lfunc.NToAs):

		#Make design matrix.  Two components: baseline and profile shape.

		M=np.ones([2,lfunc.Nbins[i]])
		M[1] = s[i]


		pnoise = lfunc.ProfileInfo[i][6]

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
		    
		    
		dNM = np.dot(lfunc.ProfileData[i], M.T)/(pnoise*pnoise)


		dNMMNM = np.dot(dNM.T, InvMNM)
		MarginLike = np.dot(dNMMNM, dNM)

		profilelike = -0.5*(logdetMNM - MarginLike)
		loglike += profilelike


		if(lfunc.getShapeletStepSize == True):
			amp = dNMMNM[1]
			for j in range(lfunc.MaxCoeff - 1):
				EvoFac = (((lfunc.psr.freqs[i]-lfunc.EvoRefFreq)/1000)**np.arange(0,lfunc.EvoNPoly+1))**2
				BVec = amp*np.roll(lfunc.InterpBasis[InterpBins[i]][:,j], -RollBins[i])
				stepsize[j] += EvoFac*np.dot(BVec, BVec)/lfunc.ProfileInfo[i][6]/lfunc.ProfileInfo[i][6]


		if(lfunc.doplot == True):
		    baseline=dNMMNM[0]
		    amp = dNMMNM[1]
		    noise = np.std(lfunc.ProfileData[i] - baseline - amp*s[i])
		    print i, amp, baseline, noise
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]), lfunc.ProfileData[i])
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]),baseline+s[i]*amp)
		    plt.show()
		    plt.plot(np.linspace(0,1,lfunc.Nbins[i]),lfunc.ProfileData[i]-(baseline+s[i]*amp))
		    plt.show()

	if(lfunc.getShapeletStepSize == True):
		for j in range(lfunc.MaxCoeff - 1):
			print "step size ", j,  stepsize[j], 1.0/np.sqrt(stepsize[j])
		return 1.0/np.sqrt(stepsize)

	return loglike


import time
start = time.clock()


for i in range(1000):
	if(i%100 == 0):
		print i



	PhaseLike(x)


end = time.clock()

print "time", start-end
