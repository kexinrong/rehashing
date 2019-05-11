import sys
import subprocess

cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file ../../../../resources/data/%s_askit.data -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file %s.knn'
query_cmd = 'mpirun -np 1 ./test_find_knn.exe  -ref_file ../../../../resources/data/%s_askit_query.data -search_all2all -glb_nref %d -eval -dim %d -k 10  -knn_file %s_query.knn'

datasets = {
	'covtype': [54, 581012, 10000],
	'shuttle': [9, 43500, 43500],
}

if __name__ == "__main__":
	ds = sys.argv[1]
	print(ds)
	d, n, m = datasets[ds]
	run_cmd = cmd %(ds, n, d, ds)
	print(run_cmd)
	subprocess.call(run_cmd.split())
	run_cmd = query_cmd %(ds, m, d, ds)
	print(run_cmd)
	subprocess.call(run_cmd.split())
