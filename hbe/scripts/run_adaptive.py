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
'susy']

cmd = './hbe conf/%s.cfg gaussian'
print("eps=0.5")

f = open('adaptive0.5.txt', 'w')
for ds in datasets:
	print(ds)
	f.write('%s\n' % ds)
	subprocess.call((cmd %(ds)).split(), stdout=f)
	f.write('\n\n\n')
f.close()