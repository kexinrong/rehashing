#!/bin/sh
#--- general options ---
machine="krakenpf"
myrundir=${SCRATCHDIR}
useremail="bo@ices.utexas.edu"
np=504    # number of MPI processes  //  96  192 8192, 4096, 16384, 12288, 24576, 49152
openmp=
walltime=00:10:00
usetimelabel=y
label=partree-test
dimension=1024
numberOfReferencePointsPerMPIProcess=10000
maxPointsPerNode=2000
GEN=7
inputFile=${SCRATCHDIR}/gabor/bin/n4100_d1008.bin
knn=2
srcdir=/nics/a/proj/csela/BoXiao/knn/src/parTree
executable="test_partree.exe"

options=" 
-rn ${numberOfReferencePointsPerMPIProcess}
-qn ${numberOfReferencePointsPerMPIProcess}
-id 10
-d ${dimension}
-gen ${GEN} 
-file ${inputFile}
-mtl 30
-mppn ${maxPointsPerNode}
-iter 3
-eval 1
-flops 1
-fr 1
-ata 1
-k ${knn}
"

 #--------- process ------
 if [ ! -f "${srcdir}/${executable}" ]; then
	  echo "CRJOB ERROR  executable ${executable} doesn't exist"
     exit 1
 fi
if [ -n "${openmp}" ]; then
   ompthreads="export OMP_NUM_THREADS=6"
   gotothreads="export GOTO_NUM_THREADS=6"
   openmpoptions="-S 1 -d 6 -ss "
	 openmplabel="WMP"
	 npRequested=$(echo "${np}*6" | bc)
else
		ompthreads=
		gotothreads=
		openmpotions=
		openmplabel="nmp"
		npRequested=$(echo "(${np}+12-${np}%12)/12 -1*(0==${np}%12)" | bc)
		npRequested=${np}
fi

mon=`date +%b`
day=`date +%d`
hrm=`date +%H%M`
if [ -n "${usetimelabel}" ]; then
   tmp=$hrm
else
   tmp=
fi
gbtag="${day}${mon}${tmp}"


label="${label}.np${np}.gen${GEN}.dim${dimension}.rn${numberOfReferencePointsPerMPIProcess}.k${knn}.mppn${maxPointsPerNode}.${openmplabel}.pe${npRequested}.${gbtag}"

OUTFILE="output-$label"

fileoptions=jobopt${label}
`touch ${fileoptions}`
echo "${options}" >> ${fileoptions}
mv ${fileoptions} ${srcdir}/results

targetExecutable="${executable}.${label}.exe"


#--------------SCRIPT begins here
echo "#PBS -A TG-ASC070050N"
echo "#PBS -j oe"
echo "#PBS -m bae"
echo "#PBS -M ${useremail}"
echo "#PBS -N ${label}"
echo "#PBS -l walltime=${walltime},size=${npRequested}"
#echo "#PBS -qdebug"

echo 'set -x'
echo 'export OMP_NUM_THREADS=1'
echo 'module load papi'

#echo 'export MPICH_COLL_SYNC=1'
echo 'export MPICH_FAST_MEMCPY=1'
echo 'export MPICH_PTL_OTHER_EVENTS=100000'
echo 'export MPICH_PTL_UNEX_EVENTS=400000'
echo 'export MPICH_PTL_MATCH_OFF=1'
echo 'export MPICH_PTL_MEMD_LIMIT=32768'
echo export MPICH_ALLTOALLVW_RECVWIN=1
echo export MPICH_ALLTOALLVW_SENDWIN=1


#echo "# code version"

echo cp ${srcdir}/results/${fileoptions} ${myrundir}
echo cp ${srcdir}/${executable} ${myrundir}/${targetExecutable}
echo cd ${myrundir}
echo date


#echo ulimit -c unlimited

targetExecutable="${executable}.${label}.exe"
echo "aprun -n ${np} ${openmpoptions}./${targetExecutable} \`cat ${fileoptions}\` >& ${OUTFILE}"

echo date

echo mv ${OUTFILE} ${srcdir}/results

