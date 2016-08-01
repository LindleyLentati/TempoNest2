def PrintEpochParams(time, string ='DMX'):

	time=30
	stoas = lfunc.psr.stoas
	mintime = stoas.min()
	maxtime = stoas.max()

	NEpochs = (maxtime-mintime)/time
	Epochs=mintime - time*0.01 + np.arange(NEpochs)*time #np.linspace(mintime-time*0.01, maxtime+time*0.01, int(NEpochs+3))
	EpochList = []
	for i in range(len(Epochs)-1):
		select_indices = np.where(np.logical_and( stoas < Epochs[i+1], stoas >= Epochs[i]))[0]
		if(len(select_indices) > 0):
			EpochList.append(i)

	EpochList = np.unique(np.array(EpochList))
	for i in range(len(EpochList)):
		if(i < 9):
			print string+"_000"+str(i+1)+" -2.90883832 0"
			print string+"R1_000"+str(i+1)+" "+str(Epochs[EpochList[i]])
			print string+"R2_000"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
			print string+"ER_000"+str(i+1)+" 0.20435656\n"

		if(i < 99 and i >= 9):
			print string+"_00"+str(i+1)+" -2.90883832 0"
			print string+"R1_00"+str(i+1)+" "+str(Epochs[EpochList[i]])
			print string+"R2_00"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
			print string+"ER_00"+str(i+1)+" 0.20435656\n"

		if(i < 999 and i >= 99):
			print string+"_0"+str(i+1)+" -2.90883832 0"
			print string+"R1_0"+str(i+1)+" "+str(Epochs[EpochList[i]])
			print string+"R2_0"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
			print string+"ER_0"+str(i+1)+" 0.20435656\n"

		if(i < 9999 and i >= 999):
			print string+"_"+str(i+1)+" -2.90883832 0"
			print string+"R1_"+str(i+1)+" "+str(Epochs[EpochList[i]])
			print string+"R2_"+str(i+1)+" "+str(Epochs[EpochList[i]+1])
			print string+"ER_"+str(i+1)+" 0.20435656\n"
