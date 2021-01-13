import core as core
import numpy as np
import sys

if __name__ == '__main__':
	#assert len(sys.argv)>1, "Need config name"
	configName = "default"
	filename = 'config/'+configName+'.ini'
	c = core.core(filename)	
	c.loadCkpt(228,True)
	c.myvalidation()
	# c.train()
	
