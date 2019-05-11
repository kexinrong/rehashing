import subprocess
import sys

datasets = {
	'acoustic': [50, 78823, 78823],
	'mnist': [784, 70000, 70000],
	'tmy': [8, 1822080, 10000],
	'covtype': [54, 581012, 10000],
	'home': [10, 928991, 100000],
	'shuttle': [9, 43500, 43500],
	'ijcnn': [22, 141691, 93067],
	'skin': [3, 245057, 100000],
	'codrna': [8, 59535, 59535],
	'corel': [32, 68040, 68040],
	'elevator': [18, 16599, 16599],
	'hep': [27, 10500000, 100000],
	'higgs': [28, 11000000, 10000],
	'housing': [8, 20640, 20640],
	'msd': [90, 463715, 10000],
	'poker': [10, 25010, 25010],
	'sensorless': [48, 58509, 58509],
	'susy': [18, 5000000, 100000],
	'cifar10': [3072, 50000, 50000],
	'aloi': [128, 108000, 9121],
	'timit': [440, 1000000, 9812],
	'glove.6B.100d': [100, 400000, 10000],
	'census': [68, 2458285, 9546],
	'svhn': [3072, 630420, 9820]
}

#for ds in ['svhn', 'census', 'glove.6B.100d', 'timit', 'aloi', 'tmy', 'covtype', 'msd']:
for ds in ['higgs']:
	cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/data/askit/%s_askit.txt -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file knn/%s.knn'
	query_cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/data/askit/%s_askit_query_clean.txt -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file knn/%s_query_clean.knn'

	f = open('knn/%s_runtime.txt' % ds, 'w')
	#for ds in datasets:
	print(ds)
	d, n, m = datasets[ds] 
	subprocess.call((cmd %(ds, n, d, ds)).split(), stdout=f)
	subprocess.call((query_cmd %(ds, m, d, ds)).split(), stdout=f)
	f.close()