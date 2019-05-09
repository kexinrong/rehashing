import subprocess

configs = ['d100r']
dim = [100]
datasets =  [1, 10, 100, 1000, 10000, 100000]

cmd = '../bin/generic %s %f %s %d'

for idx, cf in enumerate(configs):
	print(cf)
	for eps in [0.01, 0.001]:
		print(eps)
		for ds in datasets:
			subprocess.call((cmd %(ds, eps, cf, dim[idx])).split())
