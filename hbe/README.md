# Dependencies and Build Instruction
Our library uses CMake, and depends on Boost, [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) and Config4Cpp. The following instructions have been tested on Linux machines. To install Boost:
```sh
sudo apt-get install libboost-all-dev
```

We include the source codes of Eigen3 and Config4Cpp as git submodules. To get the source codes:
```sh
~/rehashing/hbe/$ git submodule init
~/rehashing/hbe/$ git submodule update
```

If you already have a recent version of Eigen3 installed on your system, you might want to use this version. The library assumes that the Eigen headers are accessible as `#include <Eigen/Dense>` etc. Alternatively, you can build Eigen from source:
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

Finally, you can build the main programs by
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

