import subprocess
import numpy as np
import sys

datasets = {
	1: [512349],
	10: [512340],
	100: [512200],
	1000: [511000],
	10000: [510000]

}
def get_relerr(c):
	est = np.loadtxt('results/%s_gen%d.txt' % (ver, c))
	exact = np.loadtxt('../../../resources/generic_gaussian/%s/exact_%d,%d.txt' % (ver, c, 500000/c), delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	print("Relative Error: %f\n" % (err / len(est)))

cmd = './askit_kde_main.exe -training_data /lfs/1/krong/hbe/resources/generic_gaussian/askit/data_%s_%d,%d.txt  -training_knn_file ../../rkdtsrc/parallelIO/knn/%s_gen%d.knn -test_data /lfs/1/krong/hbe/resources/generic_gaussian/askit/query_%s_%d,%d.txt -test_knn_file ../../rkdtsrc/parallelIO/knn/%s_query_gen%d.knn -training_N %d -test_N 10000 -output results/%s_gen%d.txt -h %f -d %d -id_tol %f'

ver = sys.argv[1]
d = int(sys.argv[2])
id_tol = float(sys.argv[3])
f = open('results/%s_gen_runtime.txt' % ver, 'w')
m = 10000
for c in [1, 10, 100, 1000, 10000, 100000]:
	print(c)
	n = 500000
	if c in datasets:
		n = datasets[c][0]
	k = 500000 / c
	# bandwidth: -0.5 / (h^2)
	h = 1.0 / np.sqrt(2)
	cmd_ds = cmd % (ver,c,k,ver,c,ver,c,k,ver,c,n,ver,c,h, d, id_tol)
	print(cmd_ds)
	subprocess.call(cmd_ds.split(), stdout=f)
	get_relerr(c)
f.close()
