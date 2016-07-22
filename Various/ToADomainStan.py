# module import
import pystan
import numpy as np
import pylab as py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from libstempo.libstempo import *
import libstempo as T


#First load pulsar.  We need the sats (separate day/second), and the file names of the archives (FNames)
psr = T.tempopulsar(parfile="../TempoNest/Examples/Example2/J0030+0451.par", timfile = "../TempoNest/Examples/Example2/J0030+0451.tim")
psr.fit()
NToAs = psr.nobs

toas=psr.toas()
residuals = np.float64(psr.residuals())
errs=psr.toaerrs*(10.0**-6)


#Check how many timing model parameters we are fitting for (in addition to phase)
numTime=len(psr.pars())+1
redChisq = psr.chisq()/(psr.nobs-len(psr.pars())-1)
TempoPriors=np.zeros([numTime,2]).astype(np.float64)

TempoPriors[0][0] = 0
TempoPriors[0][1] = 1.0/np.sqrt(np.sum(1.0/(psr.toaerrs*10.0**-6)**2)) 

for i in range(1,numTime):
        TempoPriors[i][0]=psr[psr.pars()[i-1]].val
        TempoPriors[i][1]=psr[psr.pars()[i-1]].err/np.sqrt(redChisq)
	print "fitting for: ", psr.pars()[i-1], TempoPriors[i][0], TempoPriors[i][1]


designMatrix=psr.designmatrix()
for i in range(numTime):
	designMatrix[:,i] *= TempoPriors[i][1]

designMatrix=np.float64(designMatrix)

def createfourierdesignmatrix_RED(t, nmodes, freq=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param freq: option to output frequencies
    @param Tspan: option to some other Tspan

    @return: F: fourier design matrix
    @return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = numpy.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    fqs = numpy.linspace(1/T, nmodes/T, nmodes)

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = numpy.cos(2*numpy.pi*fqs[ct]*t)
        F[:,ii+1] = numpy.sin(2*numpy.pi*fqs[ct]*t)
        ct += 1
    
    if freq:
        return F, fqs
    else:
        return F


NModes = 10
FMatrix = np.float64(createfourierdesignmatrix_RED(psr.toas(), nmodes = NModes))


 
# STAN model (this is the most important part)
regress_code = """
data {
 int<lower = 0> N; // number of observations
 int<lower = 0> T; // number of timing model parameters
 int<lower = 0> F; // number of Fourier Modes
 vector[N] y; // residuals
 vector[N] x; // BATs
 vector[N] err; //ToA Errors
 matrix[N,T] M; //Timing model design matrix
 matrix[N,2*F] FMatrix; //Fourier Basis Vector
}
parameters {
 vector[T] e; // timing model parameters
 vector[2*F] a; //Fourier Coefficients
 vector[F] P; //Prior on the Fourier Coefficients
}
transformed parameters {
 vector[N] mu; // fitted values
 vector[2*F] aScaled; //reScaled Fourier coefficients
 for (j in 1:F){
 	aScaled[j] <- a[j]*pow(10.0, P[j]);
        aScaled[j+F] <- a[j+F]*pow(10.0, P[j]);
 }
 mu <- M*e + FMatrix*aScaled; 

}
model {
 //Priors
 e ~ normal(0, 100);
 a ~ normal(0,1);
 P ~ uniform(-10,-3);
 y ~ normal(mu, err);
}
"""
 
# make a dictionary containing all data to be passed to STAN
regress_dat = {'x': toas,
 'y': residuals,
 'err': errs,
 'N': NToAs,
 'T': numTime,
 'F': NModes,
 'M': designMatrix,
 'FMatrix': FMatrix}

## Initialization function
def myinit(nChains):
    params={}
    params['e']=np.zeros(numTime)
    params['a']=np.zeros(2*NModes)
    params['P']=np.random.uniform(-10,-3,NModes)
    return [params]*nChains

nChains = 4
initVals=myinit(nChains)
# Fit the model
fit = pystan.stan(model_code=regress_code, data=regress_dat,
 iter=10000, chains=nChains, init=initVals)
 
# model summary
print fit
 
# show a traceplot of ALL parameters. This is a bear if you have many
fit.traceplot()
py.show()
 
# Instead, show a traceplot for single parameter
fit.plot(['e'])
py.show()
