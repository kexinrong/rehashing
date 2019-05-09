#export PLATFORM=ronaldo
cd ..
source sourceme
make clean
make -j12
cd binTree
make clean
make
