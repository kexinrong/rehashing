import numpy as np
import sys

def get_relerr(f1, f2):
	est = np.loadtxt('results/%s' % f1)
	exact = np.loadtxt('../../../resources/exact/%s' %f2, delimiter=',')
	err = 0
	for i in range(len(exact)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	print("Relative Error: %f\n" % (err / len(exact)))


f1 = sys.argv[1]
f2 = sys.argv[2]
get_relerr(f1, f2)