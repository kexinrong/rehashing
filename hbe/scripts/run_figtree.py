import subprocess

datasets =  ['shuttle',
'housing',
'acoustic',
'mnist',
'tmy',
'covtype',
'home',
'ijcnn',
'skin',
'codrna',
'corel',
'elevator',
'msd',
'poker',
'sensorless',
'susy']

cmd = '../bin/realdata %s %f'

for eps in [0.1, 0.05, 0.01]:
	print(eps)
	# f = open('tree.txt', 'w')
	for ds in datasets:
		print(ds)
		subprocess.call((cmd %(ds, eps)).split())
	# f.close()