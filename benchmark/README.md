We include source code and wrapper scripts for [FigTree](https://github.com/vmorariu/figtree) and [ASKIT](http://padas.ices.utexas.edu/libaskit/) for comparision and provide simple instructions on running the libraries on example datasets below. For details, please refer to the corresponding documentation of the libraries.

# FigTree
To build FigTree from source
```sh
~/rehashing/benchmark/figtree $ make
~/rehashing/benchmark/figtree $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/rehashing/benchmark/figtree/lib
~/rehashing/benchmark/figtree $ cd samples
~/rehashing/benchmark/figtree/samples $ make
```
To run FigTree on real-world datasets: ```../bin/realdata [dataset] [epsilon] [bw multiplier] [exact file]```
```sh
~/rehashing/benchmark/figtree/samples $ ../bin/realdata shuttle 0.1 1 shuttle_gaussian.txt
~/rehashing/benchmark/figtree/samples $ ../bin/realdata covtype 0.05 2 covtype_gaussian.txt
```
The smaller epsilon is, the more accurate the results will be, at the expense of increased computational complexity. Final bandwdith = sqrt(2) * multiplier * [Scott's factor](https://nicta.github.io/dora/generated/generated/scipy.stats.gaussian_kde.html#id8). The exact densities are used for computing relative errors.

FigTree contains an optimizer that chooses between four evaluation strategies: ```FIGTREE_EVAL_DIRECT```, ```FIGTREE_EVAL_DIRECT_TREE```, ```FIGTREE_EVAL_IFGT```, and ```FIGTREE_EVAL_IFGT_TREE```. FigTree only works with Gaussian kernels.

# ASKIT
