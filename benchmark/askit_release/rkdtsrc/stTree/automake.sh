#export PLATFORM=ronaldo
cd ..
source sourceme
make clean
make -j12
cd stTree
make clean
make
