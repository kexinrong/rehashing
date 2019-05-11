import subprocess
import sys

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
'susy']

ds = sys.argv[1]
cmd = './hbe conf/%s.cfg gaussian'
eps = 0.5
times = False
not_end = True

f = open('adaptive_%s.txt' % ds, 'w')
while not_end:
	print(ds)
	f.write('%s\n' % ds)
	subprocess.call((cmd %(ds)).split(), stdout=f)
	f.write('\n\n\n')
	
f.close()