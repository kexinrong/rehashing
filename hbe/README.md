# Dependencies and Build Instruction
Our library uses CMake, and depends on Boost, Eigen3 and Config4Cpp. We include Eigen3 and Config4Cpp as git submodules. To build these libraries from source:
```sh
~/rehashing/hbe/$ git submodule init
~/rehashing/hbe/$ git submodule update
```

To build Eigen
```sh
~/rehashing/hbe/$ mkdir build
~/rehashing/hbe/$ cd build
~/rehashing/hbe/build/$ cmake ../lib/eigen-git-mirror/
```
To build Config4Cpp
```sh
~/rehashing/hbe/$ cd lib/config4cpp/
~/rehashing/hbe/lib/config4cpp/$ make
```

To build the experiments
```sh
~/rehashing/hbe/$ cmake .
~/rehashing/hbe/$ make
```

# Project Structure
- ```alg/```: implementations of main algorithms, including HBS, HBE, diagnosis, baseline sketching algorithms and baseline KDE algorithms.
- ```data/```: implementations to generate various synthetic datasets, including the "worst-case" instance and "D-structure" instance described in the paper.
- ```util/```: various utility functions.
- ```main/```: main functions that depends on the HBE library.
- ```conf/```: default location for config files.
- ```scripts/```: Python wrapper scripts for the experiment.

