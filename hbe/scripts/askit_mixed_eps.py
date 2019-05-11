import subprocess
import numpy as np
import sys


def get_relerr(d):
	est = np.loadtxt('results/%s_gen%d.txt' % (ver, d))
	exact = np.loadtxt('../../../resources/syn/exact_generic%d.txt' % (d), delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	return err / len(est)

cmd = './askit_kde_main.exe -training_data /lfs/1/krong/hbe/resources/generic_gaussian/askit/data_%s_%d.txt  -training_knn_file ../../rkdtsrc/parallelIO/knn/%s_gen%d.knn -test_data /lfs/1/krong/hbe/resources/generic_gaussian/askit/query_%s_%d.txt -test_knn_file ../../rkdtsrc/parallelIO/knn/%s_query_gen%d.knn -training_N %d -test_N 10000 -output results/%s_gen%d.txt -h %f -d %d -id_tol %f'


d = int(sys.argv[1])
ver = 'mixed'
m = 10000
n = 1022345

head = 0
tail = 1
eps = 0.1
times = False
not_end = True
# bandwidth: -0.5 / (h^2)
h = 1.0 / np.sqrt(2)


while not_end:
	print(eps)
	cmd_ds = cmd % (ver,d,ver,d,ver,d,ver,d,n,ver,d,h, d, eps)
	print(cmd_ds)
	p = subprocess.Popen(cmd_ds.split(), stdout=subprocess.PIPE)
	out, _ = p.communicate()
	out = out.decode('utf-8')
	print(out.split('\n')[-2:])
	err = get_relerr(d)
	print("Relative Error: %f\n" % err)

	if eps == 0.1:
		if err > 0.1:
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
	cmd_ds = cmd % (ver,d,ver,d,ver,d,ver,d,n,ver,d,h, d, eps)
	p = subprocess.Popen(cmd_ds.split(), stdout=subprocess.PIPE)
	out, _ = p.communicate()
	out = out.decode('utf-8')
	print(out.split('\n')[-2:])
	err = get_relerr(d)
	print("Relative Error: %f\n" % err)

	if err < 0.11 and err > 0.09:
		break
	elif err > 0.1:
		tail = eps
	else:
		head = eps

