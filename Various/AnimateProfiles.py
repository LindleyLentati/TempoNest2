import numpy as np
import glob as glob
import matplotlib.pyplot as plt

root="PC-warsData-LinTime1-Amps3-Red-Stoc-4-"
flist=glob.glob(root+"*Profile.txt")

for i in flist:
	f, axarr = plt.subplots(3)

	MJD=(open(i).readlines()[0].strip('\n').split())[0][1:]

	Prof=np.loadtxt(i).T
	Phase=Prof[0]
	Data=Prof[1]
	Model=Prof[2]
	
	Res=Prof[1]-Prof[2]

	axarr[0].set_title(MJD)
	axarr[0].set_xlim([450, 600])
	axarr[0].errorbar(Phase, Data/np.max(Model), yerr=Prof[3]/np.max(Model), linestyle='')
	axarr[0].plot(Phase, Model/np.max(Model))
	axarr[0].plot(Phase, Prof[6]/np.max(Model))
	axarr[0].plot(Phase, Prof[7]/np.max(Model))
	axarr[0].plot(Phase, Prof[8]/np.max(Model))
	axarr[0].plot(Phase, Prof[9]/np.max(Model))
	axarr[0].plot(Phase, Prof[10]/np.max(Model))
	axarr[0].plot(Phase, Prof[11]/np.max(Model))
	axarr[0].plot(Phase, Prof[12]/np.max(Model))

	axarr[1].set_xlim([450, 600])
	axarr[1].set_ylim([-5, 5])
	axarr[1].errorbar(Phase, Res/Prof[3], yerr=np.ones(len(Prof[3])), linestyle='')

	axarr[2].set_xlim([0, 1024])
	axarr[2].set_ylim([-5, 5])
	axarr[2].errorbar(Phase, Res/Prof[3], yerr=np.ones(len(Prof[3])), linestyle='')

	plt.savefig('plots/'+MJD+'.png')
	plt.close()
	



