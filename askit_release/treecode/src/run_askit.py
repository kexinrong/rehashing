import subprocess
import numpy as np

datasets = {
	'test': [3, 4, 4, 1]
}

def get_relerr(ds):
	est = np.loadtxt('results/%s.txt' % ds)
	exact = np.loadtxt('../../../resources/exact/%s_gaussian.txt' % ds, delimiter=',')
	err = 0
	for i in range(len(est)):
		err += np.abs(est[i] - exact[i][0]) / exact[i][0]
	print("Relative Error: %f\n" % (err / len(est)))

cmd = './askit_kde_main.exe -training_data ../../../resources/data/askit/%s_askit.txt  -training_knn_file ../../rkdtsrc/parallelIO/knn/%s.knn -test_data ../../../resources/data/askit/%s_askit_query.txt -test_knn_file ../../rkdtsrc/parallelIO/knn/%s_query.knn -training_N %d -test_N %d -d %d -output results/%s.txt -h %f'

f = open('results/runtime.txt', 'w')
for ds in datasets:
	print(ds)
	f.write('%s\n' % ds)
	d, n, m, h = datasets[ds] 
	# bandwidth: -0.5 / (h^2)
	h = h / np.sqrt(2)
	cmd_ds = cmd % (ds, ds, ds, ds, n, m, d, ds, h)
	subprocess.call(cmd_ds.split(), stdout=f)
	f.write('\n\n\n')
	get_relerr(ds)
f.close()