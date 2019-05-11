import subprocess
import numpy as np
import sys

datasets = {
	'acoustic': [50, 78823, 78823, 1.14772],
	'mnist': [784, 70000, 70000, 6.97166],
	'tmy': [8, 1822080, 100000, 0.425403],
	'covtype': [54, 581012, 581012, 2.2499],
	'home': [10, 928991, 100000, 0.529942],
	'shuttle': [9, 43500, 43500, 0.621882],
	'ijcnn': [22, 141691, 100000, 0.896162],
	'skin': [3, 245057, 100000, 0.240226],
	'codrna': [8, 59535, 59535, 0.565741],
	'corel': [32, 68040, 68040, 1.03818],
	'elevator': [18, 16599, 16599, 1.81854],
	'housing': [8, 20640, 20640, 0.617954],
	'msd': [90, 463715, 100000, 6.15468],
	'poker': [10, 25010, 25010, 0.686063],
	'sensorless': [48, 58509, 58509, 2.29017],
	#'hep': [27, 10500000, 100000, 7.07107],
	#'higgs': [28, 11000000, 100000, 7.07107],
	#'susy': [18, 5000000, 100000, 5.65685]
}

def get_relerr(ds):
	est = np.loadtxt('results/%s.txt' % ds)
	exact = np.loadtxt('../../../resources/exact/%s_gaussian.txt' % ds, delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	print("Relative Error: %f\n" % (err / len(est)))

cmd = './askit_kde_main.exe -id_tol %f -training_data /lfs/1/krong/hbe/resources/data/askit/%s_askit.txt  -training_knn_file ../../rkdtsrc/parallelIO/knn/%s.knn -test_data /lfs/1/krong/hbe/resources/data/askit/%s_askit_query.txt -test_knn_file ../../rkdtsrc/parallelIO/knn/%s_query.knn -training_N %d -test_N %d -d %d -output results/%s.txt -h %f'

eps = float(sys.argv[1])
print(eps)
f = open('results/runtime_%f.txt' % eps, 'w')
for ds in datasets:
	print(ds)
	f.write('%s\n' % ds)
	d, n, m, h = datasets[ds]
	# bandwidth: -0.5 / (h^2)
	h = h / np.sqrt(2)
	cmd_ds = cmd % (eps, ds, ds, ds, ds, n, m, d, ds, h)
	subprocess.call(cmd_ds.split(), stdout=f)
	f.write('\n')
	get_relerr(ds)
f.close()
