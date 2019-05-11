import subprocess
import sys
import numpy as np

datasets = {
	'covtype': [54, 581012, 10000, 2.2499],
	'shuttle': [9, 43500, 43500, 0.621882],

}

def get_relerr(ds):
	est = np.loadtxt('results/%s.txt' % ds)
	exact = np.loadtxt('../../../../resources/exact/%s_gaussian.txt' % ds, delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	print("Relative Error: %f\n" % (err / len(est)))

cmd = './askit_kde_main.exe -id_tol %f -training_data ../../../../resources/data/%s_askit.data  -training_knn_file ../../rkdtsrc/parallelIO/%s.knn -test_data ../../../../resources/data/%s_askit_query.data -test_knn_file ../../rkdtsrc/parallelIO/%s_query.knn -training_N %d -test_N %d -d %d -output results/%s.txt -h %f'

if __name__ == "__main__":
	ds = sys.argv[1]
	eps = float(sys.argv[2])
	print(ds)
	d, n, m, h = datasets[ds]
	# bandwidth: -0.5 / (h^2)
	h = h / np.sqrt(2)
	cmd_ds = cmd % (eps, ds, ds, ds, ds, n, m, d, ds, h)
	print(cmd_ds)
	subprocess.call(cmd_ds.split())
	get_relerr(ds)