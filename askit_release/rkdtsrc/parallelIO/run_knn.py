import subprocess

datasets = {
	'test': [3, 4, 4, 1]
}

cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file ../../../resources/data/askit/%s_askit.txt -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file knn/%s.knn'

f = open('knn/runtime.txt', 'w')
for ds in datasets:
	print(ds)
	f.write('%s\n' % ds)
	d, n, m = datasets[ds] 
	subprocess.call((cmd %(ds, n, d, ds)).split(), stdout=f)
	f.write('\n\n\n')
f.close()