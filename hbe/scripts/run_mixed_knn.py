import subprocess
import sys


N = 1022345
cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/generic_gaussian/askit/data_%s_%d.txt -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file knn/%s_gen%d.knn'
query_cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/generic_gaussian/askit/query_%s_%d.txt -search_all2all -glb_nref 10000 -eval -dim %d -k 10  -knn_file knn/%s_query_gen%d.knn'

ver = 'mixed'
f = open('knn/generic_runtime_%s.txt' % ver, 'w')
m = 10000
for d in [5, 10, 20, 50, 100, 200, 500, 1000]:
	print(d)
	n = N
	subprocess.call((cmd %(ver, d, n, d, ver, d)).split(), stdout=f)
	subprocess.call((query_cmd %(ver, d, d, ver, d)).split(), stdout=f)
f.close()
