#PBS -q q
#PBS -l nodes=1
#PBS -l walltime=00:05:00
#PBS -N work
cd /nethome/bxiao3/src/knn/mpi_repartition
OMPI_MCA_mpi_yield_when_idle=0
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openmpi-1.4.3-icc/lib/
/opt/openmpi-1.4.3-icc/bin/mpirun --hostfile $PBS_NODEFILE -np 5 ./test

