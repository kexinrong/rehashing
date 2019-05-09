import subprocess
import sys


cmd = '../bin/gen100 %s %f %s %d'
ds = sys.argv[1]
path = sys.argv[2]
eps = 0.01
head = 0
tail = 1
times = False
not_end = True

while not_end:
	p = subprocess.Popen((cmd %(ds, eps, path, 100)).split(), stdout=subprocess.PIPE)
	out, err = p.communicate()
	out = out.decode('utf-8')
	print(out)
	err = float((out.split('\n')[-3]).split(':')[1])
	
	if eps == 0.01:
		if err > 0.1:
			eps /= 2
		elif err < 0.1:
			eps *= 2
			times = True
	else:
		if times and err > 0.1:
			head = eps / 2
			tail = eps
			not_end = False
		elif not times and err < 0.1:
			head = eps
			tail = eps * 2
			not_end = False
		if times:
			eps *= 2
		else:
			eps /= 2

print("Binary search: [%f, %f]" % (head, tail))
while True:
	eps = (head + tail) / 2
	p = subprocess.Popen((cmd %(ds, eps, path, 100)).split(), stdout=subprocess.PIPE)
	out, err = p.communicate()
	out = out.decode('utf-8')
	print(out)
	err = float((out.split('\n')[-3]).split(':')[1])

	if err < 0.11 and err > 0.09:
		break
	elif err > 0.1:
		tail = eps
	else:
		head = eps


