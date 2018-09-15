#export PLATFORM=ronaldo
cd ..
source sourceme
make clean
make -j12
cd oldtree
make clean
make
