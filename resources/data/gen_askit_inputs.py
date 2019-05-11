# Generate ASKIT input files from data file and exact file

import sys
import numpy as np

def output(fname, a):
    f = open(fname, 'w')
    for i in range(a.shape[0]):
        s = []
        for j in range(a.shape[1]):
            if a[i][j] == 0:
                s.append('0')
            else:
                s.append('%.8f ' % a[i][j])
        f.write(' '.join(s))
        f.write('\n')
    f.close()

if __name__ == "__main__":
    ds = sys.argv[1]
    data_file = sys.argv[2]
    exact_file = sys.argv[3]

    data = np.loadtxt('%s' % data_file, delimiter=',')
    # Output space seperate file
    output('%s_askit.data' % ds, data)

    e = np.loadtxt('../exact/%s' % exact_file, delimiter=',')
    # Get query data points
    query = np.zeros((e.shape[0], data.shape[1]))
    for i in range(e.shape[0]):
        query[i] = data[int(e[i][1])]
    # Output space seperate file
    output('%s_askit_query.data' % ds, query)

