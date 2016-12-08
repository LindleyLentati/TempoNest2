def PreComputeFFTShapelets(interpTime = 1, MeanBeta = 0.1, ToPickle = False, FromPickle = False):


	print("Calculating Shapelet Interpolation Matrix : ", interpTime, MeanBeta);

	'''
	/////////////////////////////////////////////////////////////////////////////////////////////  
	/////////////////////////Profile Params//////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////
	'''

	InterpBins = np.max(lfunc.Nbins)

	numtointerpolate = np.int(lfunc.ReferencePeriod/InterpBins/interpTime/10.0**-9)+1
	InterpolatedTime = lfunc.ReferencePeriod/InterpBins/numtointerpolate
	lfunc.InterpolatedTime  = InterpolatedTime	



	lenRFFT = len(np.fft.rfft(np.ones(InterpBins)))

	interpStep = 1.0/InterpBins/numtointerpolate


	if(FromPickle == False):


                InterpFShapeMatrix = np.zeros([numtointerpolate, lenRFFT, lfunc.MaxCoeff])+0j
                InterpFJitterMatrix = np.zeros([numtointerpolate,lenRFFT, lfunc.MaxCoeff])+0j


		for t in range(numtointerpolate):

			lfunc.update_progress(np.float64(t)/numtointerpolate)

			binpos = t*interpStep

			rfftfreqs=np.linspace(0,0.5*InterpBins,0.5*InterpBins+1)
			FullMatrix = np.ones([np.sum(lfunc.MaxCoeff+1), len(2*np.pi*rfftfreqs)])
			FullCompMatrix = np.zeros([np.sum(lfunc.MaxCoeff+1), len(2*np.pi*rfftfreqs)]) + 0j
			FullJitterMatrix = np.zeros([np.sum(lfunc.MaxCoeff), len(2*np.pi*rfftfreqs)]) + 0j

			ccount = 0
			for comp in range(lfunc.fitNComps):



				if(lfunc.MaxCoeff > 1):
					lfunc.TNothpl(lfunc.MaxCoeff+1, 2*np.pi*rfftfreqs*MeanBeta, FullMatrix[ccount:ccount+lfunc.MaxCoeff+1])

				ExVec = np.exp(-0.5*(2*np.pi*rfftfreqs*MeanBeta)**2)
				FullMatrix[ccount:ccount+lfunc.MaxCoeff+1]=FullMatrix[ccount:ccount+lfunc.MaxCoeff+1]*ExVec

				for coeff in range(lfunc.MaxCoeff+1):
					FullCompMatrix[ccount+coeff] = FullMatrix[ccount+coeff]*(1j**coeff)

				rollVec = np.exp(2*np.pi*((binpos+0.5)*InterpBins)*rfftfreqs/InterpBins*1j)


				ScaleFactors = lfunc.Bconst(MeanBeta, np.arange(lfunc.MaxCoeff+1))




				FullCompMatrix[ccount:ccount+lfunc.MaxCoeff+1] *= rollVec
				for i in range(lfunc.MaxCoeff+1):
					FullCompMatrix[i+ccount] *= ScaleFactors[i]


				ccount+=lfunc.MaxCoeff+1

	
			

			FullJitterMatrix[0] = (1.0/np.sqrt(2.0))*(-1.0*FullCompMatrix[1])*MeanBeta
			for i in range(1,lfunc.MaxCoeff):
				FullJitterMatrix[i] = (1.0/np.sqrt(2.0))*(np.sqrt(1.0*i)*FullCompMatrix[i-1] - np.sqrt(1.0*(i+1))*FullCompMatrix[i+1])*MeanBeta


			FullCompMatrix = FullCompMatrix.T
			FullJitterMatrix = FullJitterMatrix.T


			InterpFShapeMatrix[t]  = np.copy(FullCompMatrix[:,:lfunc.MaxCoeff])
			InterpFJitterMatrix[t] = np.copy(FullJitterMatrix[:,:lfunc.MaxCoeff])


		threshold = 10.0**-10
		upperindex=1
		while(np.max(np.abs(InterpFShapeMatrix[0,upperindex:,:])) > threshold):
			upperindex += 5
			if(upperindex >= lenRFFT):
				upperindex = lenRFFT-1
				break
			print "upper index is:", upperindex,np.max(np.abs(InterpFShapeMatrix[0,upperindex:,:]))
		#InterpShapeMatrix = np.array(InterpShapeMatrix)
		#InterpJitterMatrix = np.array(InterpJitterMatrix)
		print("\nFinished Computing Interpolated Profiles")


		lfunc.InterpFBasis = InterpFShapeMatrix[:,1:upperindex]
		lfunc.InterpFJitterMatrix = InterpFJitterMatrix[:,1:upperindex]
		lfunc.InterpolatedTime  = InterpolatedTime


		Fdata =  np.fft.rfft(lfunc.ProfileData, axis=1)[:,1:upperindex]

		lfunc.NFBasis = upperindex - 1
		lfunc.ProfileFData = np.zeros([lfunc.NToAs, 2*lfunc.NFBasis])
		lfunc.ProfileFData[:, :lfunc.NFBasis] = np.real(Fdata)
		lfunc.ProfileFData[:, lfunc.NFBasis:] = np.imag(Fdata)
	
                if(ToPickle == True):
                        print "\nPickling Basis"
                        output = open(lfunc.root+'-TScrunch.Basis.pickle', 'wb')
                        pickle.dump(lfunc.ProfileFData, output)
                        pickle.dump(lfunc.InterpFJitterMatrix, output)
			pickle.dump(lfunc.InterpFBasis, output)
                        output.close()

        if(FromPickle == True):
                print "Loading Basis from Pickled Data"
                pick = open(lfunc.root+'-TScrunch.Basis.pickle', 'rb')
                lfunc.ProfileFData = pickle.load(pick)
                lfunc.InterpFJitterMatrix  = pickle.load(pick)
		lfunc.InterpFBasis = pickle.load(pick)
                pick.close()
		lfunc.NFBasis = np.shape(lfunc.InterpFBasis)[1]
		print "Loaded NFBasis: ", lfunc.NFBasis


