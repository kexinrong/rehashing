import subprocess

datasets = {
	'acoustic': [50, 78823, 78823],
	'mnist': [784, 70000, 70000],
	'tmy': [8, 1822080, 100000],
	'covtype': [54, 581012, 581012],
	'home': [10, 928991, 100000],
	'shuttle': [9, 43500, 43500],
	'ijcnn': [22, 141691, 100000],
	'skin': [3, 245057, 100000],
	'acoustic': [50, 78823, 78823],
	'codrna': [8, 59535, 59535],
	'corel': [32, 68040, 68040],
	'elevator': [18, 16599, 16599],
	'hep': [27, 10500000, 100000],
	'higgs': [28, 11000000, 100000],
	'housing': [8, 20640, 20640],
	'msd': [90, 463715, 100000],
	'poker': [10, 25010, 25010],
	'sensorless': [48, 58509, 58509],
	'susy': [18, 5000000, 100000]
}

cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/data/askit/%s_askit.txt -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file knn/%s.knn'

f = open('knn/runtime.txt', 'w')
for ds in datasets:
	print(ds)
	f.write('%s\n' % ds)
	d, n, m = datasets[ds] 
	subprocess.call((cmd %(ds, n, d, ds)).split(), stdout=f)
	f.write('\n\n\n')
f.close()