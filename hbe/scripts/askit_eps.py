import subprocess
import numpy as np
import sys

datasets = {
	'acoustic': [50, 78823, 78823, 1.14772],
	'mnist': [784, 70000, 70000, 6.97166],
	'tmy': [8, 1822080, 10000, 0.425403],
	'covtype': [54, 581012, 10000, 2.2499],
	'home': [10, 928991, 100000, 0.529942],
	'shuttle': [9, 43500, 43500, 0.621882],
	'ijcnn': [22, 141691, 93067, 0.896162],
	'skin': [3, 245057, 100000, 0.240226],
	'codrna': [8, 59535, 59535, 0.565741],
	'corel': [32, 68040, 68040, 1.03818],
	'elevator': [18, 16599, 16599, 1.81854],
	'housing': [8, 20640, 20640, 0.617954],
	'msd': [90, 463715, 10000, 6.15468],
	'poker': [10, 25010, 25010, 0.686063],
	'sensorless': [48, 58509, 58509, 2.29017],
	'cifar10': [3072, 50000, 50000, 1],
	'aloi': [128, 108000, 9121, 1],
	'timit': [440, 1000000, 9812, 1],
	'glove.6B.100d': [100, 400000, 10000, 1],
	'census': [68, 2458285, 9546, 1],
	'svhn': [3072, 630420, 9820, 1],
	'higgs': [28, 11000000, 100000, 7.07107],
	#'hep': [27, 10500000, 100000, 7.07107],
	#'susy': [18, 5000000, 100000, 5.65685]
}

def get_relerr(ds):
	est = np.loadtxt('results/%s_%s.txt' % (ds, suffix))
	exact = np.loadtxt('../../../resources/exact/%s_gaussian%s.txt' % (ds, suffix), delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	return err / len(est)

cmd = './askit_kde_main.exe -id_tol %f -training_data /lfs/1/krong/hbe/resources/data/askit/%s_askit.txt  -training_knn_file ../../rkdtsrc/parallelIO/knn/%s.knn -test_data /lfs/1/krong/hbe/resources/data/askit/%s_askit_query%s.txt -test_knn_file ../../rkdtsrc/parallelIO/knn/%s_query%s.knn -training_N %d -test_N %d -d %d -output results/%s_%s.txt -h %f'

ds = sys.argv[1]
h = float(sys.argv[2])
suffix = sys.argv[3]
# bandwidth: -0.5 / (h^2)
h = h / np.sqrt(2)
head = 0
tail = 1
eps = 1
times = False
not_end = True
d, n, m, _ = datasets[ds]

while not_end:
	print(eps)
	# bandwidth: -0.5 / (h^2)
	cmd_ds = cmd % (eps, ds, ds, ds, suffix, ds, suffix, n, m, d, ds, suffix, h)
	print(cmd_ds)
	p = subprocess.Popen(cmd_ds.split(), stdout=subprocess.PIPE)
	out, _ = p.communicate()
	out = out.decode('utf-8')
	print(out.split('\n')[-2:])
	err = get_relerr(ds)
	print("Relative Error: %f\n" % err)

	if eps == 1:
		if err > 1:
			eps /= 2
		elif err < 0.1:
			eps *= 2
			times = True
	else:
		if times and err > 0.1:
			head = eps / 2
			tail = eps
			not_end = False
		elif not times and err < 0.1:
			head = eps
			tail = eps * 2
			not_end = False
		if times:
			eps *= 2
		else:
			eps /= 2


print("Binary search: [%f, %f]" % (head, tail))
while True:
	eps = (head + tail) / 2
	print(eps)
	cmd_ds = cmd % (eps, ds, ds, ds, suffix, ds, suffix, n, m, d, ds, suffix, h)
	print(cmd_ds)
	p = subprocess.Popen(cmd_ds.split(), stdout=subprocess.PIPE)
	out, _ = p.communicate()
	out = out.decode('utf-8')
	print(out.split('\n')[-2:])
	err = get_relerr(ds)
	print("Relative Error: %f\n" % err)

	if err < 0.11 and err > 0.09:
		break
	elif err > 0.1:
		tail = eps
	else:
		head = eps


