import subprocess
import numpy as np

datasets = {
	'poker': [10, 25010, 25010, 2.82843],
	'shuttle': [9, 43500, 43500, 1.41421],
	'codrna': [8, 59535, 59535, 0.707107],
	'housing': [8, 20640, 20640, 1.41421],
	'acoustic': [50, 78823, 78823, 1.41421],
	'mnist': [784, 70000, 70000, 14.1421],
	'tmy': [8, 1822080, 100000, 2.82843],
	'covtype': [54, 581012, 581012, 2.81237],
	'home': [10, 928991, 100000, 0.848528],
	'ijcnn': [22, 141691, 100000, 2.12132],
	'skin': [3, 245057, 100000, 0.282843],
	'corel': [32, 68040, 68040, 2.12132],
	'elevator': [18, 16599, 16599, 2.12132],
	'msd': [90, 463715, 100000, 4.24264],
	'sensorless': [48, 58509, 58509, 2.82843]#,
	#'hep': [27, 10500000, 100000, 7.07107],
	#'higgs': [28, 11000000, 100000, 7.07107],
	#'susy': [18, 5000000, 100000, 5.65685]
}

def get_relerr(ds):
	est = np.loadtxt('results/%s_cube.txt' % ds)
	exact = np.loadtxt('/lfs/1/krong/hbe/resources/exact/%s_gaussian.txt' % ds, delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	print("Relative Error: %f\n" % (err / len(est)))

cmd = './kde --references_in=/lfs/1/krong/hbe/resources/data/clean/%s.csv --queries_in=/lfs/1/krong/hbe/resources/data/clean/%s_query.csv --densities_out=results/%s_cube.txt --kernel=gaussian  --series_expansion_type=hypercube --relative_error=0.1 --probability 0.99 --bandwidth=%f'

f = open('multi_runtime.txt', 'w')
for ds in datasets:
	print(ds)
	d, n, m, h = datasets[ds] 
	# bandwidth: -0.5 / (h^2)
	h = h / np.sqrt(2)
	cmd_ds = cmd % (ds, ds, ds, h)
	print(cmd_ds)
	subprocess.call(cmd_ds.split())
	#subprocess.call(cmd_ds.split(), stdout=f)
	get_relerr(ds)
f.close()