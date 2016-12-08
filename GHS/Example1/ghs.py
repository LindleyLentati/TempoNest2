from ctypes import cdll
import sys
from ctypes import *
from numpy.ctypeslib import as_array
import signal
import inspect
import ctypes

libname = 'libghs'

libname += {
	'darwin' : '.dylib',
	'win32'  : '.dll',
	'cygwin' : '.dll',
}.get(sys.platform, '.so')


lib = cdll.LoadLibrary(libname)
	
c_double_p=ctypes.POINTER(ctypes.c_double)



def run_guided_hmc( neg_logpost,
		write_extract,
		num_dim, 
		start_point,
		step_sizes,
		file_prefix,
		dim_scale_fact = 0.4,
		max_steps = 10,
		seed = -1,
		resume = 1,
		feedback_int = 100,
		nburn = 1000,
		nsamp = 10000,
		doMaxLike = 0):

	"""
	Runs the GHS
		
	"""

	loglike_type = CFUNCTYPE(None, POINTER(c_int), POINTER(c_double),
		POINTER(c_double), POINTER(c_double))

	write_extract_type  = CFUNCTYPE(None, POINTER(c_int), POINTER(c_double),
		POINTER(c_double), POINTER(c_double))
		
	c_double_p=ctypes.POINTER(ctypes.c_double)
		
	lib.run_guided_hmc( c_int(num_dim),
			start_point.ctypes.data_as(c_double_p),
			c_double(dim_scale_fact),
			c_int(max_steps),
			step_sizes.ctypes.data_as(c_double_p),
			create_string_buffer(file_prefix.encode(),100),
			c_int(seed),
			c_int(resume),
			c_int(feedback_int),
			loglike_type(neg_logpost),
			write_extract_type(write_extract),
			c_int(nburn),
			c_int(nsamp),
			c_int(doMaxLike))



