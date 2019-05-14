## Dependencies and Build Instruction
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

## Project Structure
- ```alg/```: implementations of main algorithms, including HBS, HBE, diagnosis, the adaptive sampling algorithm, baseline sketching algorithms and baseline KDE algorithms. Please refer to the [documentation](https://kexinrong.github.io/rehashing/hierarchy.html) for details of each class.
- ```data/```: implementations to generate various synthetic datasets, including the "worst-case" instance and "D-structure" instance described in the paper.
- ```util/```: various utility functions.
- ```conf/```: default location for config files. 
- ```main/```: main programs that depends on the HBE library.

## Examples 
In ```main/```, we provide a few example main programs that uses the HBE library. To start, specify the input dataset, KDE parameters (kernel type, bandwidth), error tolerance (\tau and \epsilon) via a config file. See details in ```conf/```. 

With the config file, we can exhaustively compute the KDE for the specified setup. Uncomment ```add_executable(hbe main/ComputeExact.cpp)``` in ```CMakeLists.txt``` to build the executable. The exact densities can be used to evaluate the accuracy of different approximation algorithms. 

#### Diagnosis 
Output the estimated relative variance of HBE and RS given dataset and hashing scheme. This can be used to compare the sampling efficiency of HBE and RS before committing to either of the algorithms for the given dataset. Uncomment ```add_executable(hbe main/Diagnosis.cpp)``` in ```CMakeLists.txt``` and build the executable.


#### Adaptive sampling 
Run the adaptive sampling algorithm given 1) dataset 2) epsilon 3) RS or HBE. Uncomment ```add_executable(hbe main/RunAdaptive.cpp)``` in ```CMakeLists.txt``` to build the executable. 

The first example below runs adaptive sampling with RS, with epsilon=0.2. The second example below runs adaptive sampling with HBE, with epsilon=0.9.
```sh
~/rehashing/hbe/$ ./hbe conf/shuttle.cfg gaussian 0.2 true
~/rehashing/hbe/$ ./hbe conf/shuttle.cfg gaussian 0.9
```

#### Sketching
Compare the relative error of Uniform, HBS, Herding and SKA under varying sketch sizes. Uncomment ```add_executable(hbe main/SketchBench.cpp)``` in ```CMakeLists.txt``` to build the executable. 
