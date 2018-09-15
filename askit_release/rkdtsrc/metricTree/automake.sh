export PLATFORM=ronaldo
cd ..
source sourceme
make clean
make -j12
cd metricTree
make clean
make
