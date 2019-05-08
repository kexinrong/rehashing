# Hashing-Based-Estimators (HBE)
HBE is a C++ library for fast kernel evaluation for high-dimensional data that also includes a python implementation for illustration purposes. HBE uses [Locality Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) (LSH) to produce *provably accurate* estimates of the [kernel density](https://en.wikipedia.org/wiki/Kernel_density_estimation#Definition) for a given query point as well as weighted generalizations thereof. HBE is designed for [radially](https://en.wikipedia.org/wiki/Radial_basis_function) decreasing kernel functions (e.g. [Gaussian](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) or [Exponential](http://www.jmlr.org/papers/volume2/genton01a/genton01a.pdf) kernels) and truly high dimensional data where [FIGTree](http://users.umiacs.umd.edu/~morariu/figtree/) and [Dual-Tree algorithms](http://jmlr.org/papers/volume16/curtin15a/curtin15a.pdf) for fast kernel evaluation are slow. 

Currently, HBE supports one LSH family: [Eucledian LSH](http://mlwiki.org/index.php/Euclidean_LSH), introduced by [(Datar, Immorlica, Indyk, Mirrokni SoCG'04)](https://dl.acm.org/citation.cfm?id=997857) in the context of solving the Nearest Neighbor Search problem for the Euclidean distance metric (see also the [E2LSH](https://www.mit.edu/~andoni/LSH/) package by [Alex Andoni](https://www.mit.edu/~andoni/)). 

# How to use HBE

The first step to use HBE is to consult the [python demo](https://github.com/stanford-futuredata/hbe/wiki/Python-Demo) that describes the tuning process. Alternatively you can directly consult the C++ documentation. In our [Wiki](https://github.com/stanford-futuredata/hbe/wiki) you can also find descriptions of how to construct tunable [synthetic benchmarks](https://github.com/stanford-futuredata/hbe/wiki/Synthetic-benchmarks) for kernel evaluation as well as produce dataset [visualizations](https://github.com/stanford-futuredata/hbe/wiki/Visualizations).

# How fast is HBE? 

The speed of HBE depends on the desired relative error, the kernel and the dataset. For relative error 0.1 under the Gaussian kernel and datasets around 1M points HBE takes less than *25ms per query* and very often around ~2ms. For specific datasets and comparison with competing methods ([FIGTree](http://users.umiacs.umd.edu/~morariu/figtree/), [ASKIT](http://padas.ices.utexas.edu/libaskit/), Random Sampling (RS)) see Table 1. 

![Table 1](https://github.com/stanford-futuredata/hbe/blob/clean/experiments/experiments_hbe.png "Table 1")

# Authors 

HBE is mainly developed by [Kexin Rong](https://kexinrong.github.io/) and [Paris Siminelakis](https://web.stanford.edu/~psimin/). HBE has grown out of research projects with our collaborators [Peter Bailis](http://www.bailis.org/), [Moses Charikar](https://engineering.stanford.edu/people/moses-charikar) and [Philip Levis](http://csl.stanford.edu/~pal/).

If you want to cite HBE in a publication, here is the bibliographic information of our research papers where the algorithms are described and analyzed:

> **Rehashing Kernel Evaluation in High Dimensions**. Paris Siminelakis, Kexin Rong, Peter Bailis, Moses Charikar, Phillip Levis. 
> *ICML 2019*

> **Hashing-Based-Estimators for Kernel Density in High Dimensions**. Moses Charikar, Paris Siminelakis, *FOCS 2017*.

# License

HBE is available under the MIT License (see LICENSE.txt)
