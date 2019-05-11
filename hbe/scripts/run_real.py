import subprocess

datasets =  ['acoustic',
'mnist',
'tmy',
'covtype',
'home',
'shuttle',
'ijcnn',
'skin',
'codrna',
'corel',
'elevator',
'housing',
'msd',
'poker',
'sensorless',
'susy',
'hep',
'higgs']

cmd = './hbe conf/%s.cfg exp'

for ds in datasets:
	print(ds)
	subprocess.call((cmd %(ds)).split())
