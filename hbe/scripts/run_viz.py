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
'census',
'timit',
'svhn',
'aloi',
'glove.6b.100d',
'hep',
'higgs']

cmd = './hbe conf/%s.cfg gaussian'

for ds in datasets:
	print(ds)
	f = open('viz/%s.txt' % ds, 'w')
	subprocess.call((cmd %(ds)).split(), stdout=f)
	f.close()

	