We include source code and wrapper scripts for [FigTree](https://github.com/vmorariu/figtree) and [ASKIT](http://padas.ices.utexas.edu/libaskit/) for comparision, and provide basic instructions on running the libraries on example datasets below. For details, please refer to the corresponding documentation of the libraries.

## FigTree
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

## ASKIT
#### Notes
The compiler flags are currently set for Intel compilers. A non-exhaustive dependency list of ASKIT include: Intel MKL, MPI, OpenMP. Input datasets to ASKIT must be space-seperated and contain no headers. To measure performance on a single core, set number of threads to 1 by ```export OMP_NUM_THREADS=1```.
#### Build
To use ASKIT, we first need to build the RKDT (Randomized KD-trees) library in ```askit_release/rkdtsrc``` (see ```askit_release/rkdtsrc/README``` for instructions). If everything builds succesfully, you should be able to see ```libcmd.a  libknn.a  librrot.a``` under ```askit_release/rkdtsrc/build```, and ```test_find_knn.exe``` under ```askit_release/rkdtsrc/parallelIO```.

After building RKDT, we can proceed to build the ASKIT library in ```askit_release/treecode``` (see ```askit_release/treecode/README``` for instructions). If everything builds succesfully, you should be able to see ```libaskit.a``` under ```askit_release/treecode/build```, and ```test_askit_main.exe``` under ```askit_release/treecode/src```.
#### Example Usage
Here we provide an example running ASKIT on the covtype dataset.
First, we generate input files to ASKIT using the helper script: ```python gen_askit_inputs.py [dataset name] [dataset] [exact file]```. The exact file should contain query density and index of the query point in the dataset in each line.
```sh
~/rehashing/resources/data $ python gen_askit_inputs.py covtype covtype.csv covtype_gaussian.txt
```
We then perform preprocessing by computing KNN on input files with the RKDT library:
```sh
~/rehashing/benchmark/askit_release/rkdtsrc/parallelIO $ python run_knn.py covtype
```
Finally, we can run ASKIT on the input files and KNN results. The second input parameter to ```run_askit.py``` controls result quality - a smaller epsilon will lead to more accurate results at the cost of longer runtime.
```sh
~/rehashing/benchmark/askit_release/treecode/src $ python run_askit.py covtype 0.5
```
