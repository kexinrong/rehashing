import subprocess
import sys

datasets = {
	1: [512349],
    10: [512340],
    100: [512200],
    1000: [511000],
    10000: [510000]}

N = 500000
cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/generic_gaussian/askit/data_%s_%d,%d.txt -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file knn/%s_gen%d.knn'
query_cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file /lfs/1/krong/hbe/resources/generic_gaussian/askit/query_%s_%d,%d.txt -search_all2all -glb_nref 10000 -eval -dim %d -k 10  -knn_file knn/%s_query_gen%d.knn'

ver = sys.argv[1]
d = int(sys.argv[2])
f = open('knn/generic_runtime_%s.txt' % ver, 'w')
m = 10000
for c in [1, 10, 100, 1000, 10000, 100000]:
	print(c)
	n = N
	if c in datasets:
		n = datasets[c][0]
	subprocess.call((cmd %(ver, c, N/c, n, d, ver, c)).split(), stdout=f)
	subprocess.call((query_cmd %(ver, c, N/c, d, ver, c)).split(), stdout=f)
f.close()
