
/**
  @file parUtils.txx
  @brief Definitions of the templated functions in the par module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
 */

#include "binUtils.h"
#include "seqUtils.h"
#include "dtypes.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include "dendro.h"

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

namespace par {

  template <typename T>
    inline int Mpi_Isend(T* buf, int count, int dest, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Isend(buf, count, par::Mpi_datatype<T>::value(),
          dest, tag, comm, request);

      return 1;

    }

  template <typename T>
    inline int Mpi_Issend(T* buf, int count, int dest, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Issend(buf, count, par::Mpi_datatype<T>::value(),
          dest, tag, comm, request);

      return 1;

    }

  template <typename T>
    inline int Mpi_Recv(T* buf, int count, int source, int tag,
        MPI_Comm comm, MPI_Status* status) {

      MPI_Recv(buf, count, par::Mpi_datatype<T>::value(),
          source, tag, comm, status);

      return 1;

    }

  template <typename T>
    inline int Mpi_Irecv(T* buf, int count, int source, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Irecv(buf, count, par::Mpi_datatype<T>::value(),
          source, tag, comm, request);

      return 1;

    }

  template <typename T, typename S>
    inline int Mpi_Sendrecv( T* sendBuf, int sendCount, int dest, int sendTag,
        S* recvBuf, int recvCount, int source, int recvTag,
        MPI_Comm comm, MPI_Status* status) {
      PROF_PAR_SENDRECV_BEGIN

        MPI_Sendrecv(sendBuf, sendCount, par::Mpi_datatype<T>::value(), dest, sendTag,
            recvBuf, recvCount, par::Mpi_datatype<S>::value(), source, recvTag, comm, status);

      PROF_PAR_SENDRECV_END
    }

  template <typename T>
    inline int Mpi_Scan( T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_SCAN_BEGIN

        MPI_Scan(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

      PROF_PAR_SCAN_END
    }

  template <typename T>
    inline int Mpi_Allreduce(T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALLREDUCE_BEGIN

        MPI_Allreduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

      PROF_PAR_ALLREDUCE_END
    }

  template <typename T>
    inline int Mpi_Alltoall(T* sendbuf, T* recvbuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALL_BEGIN

        MPI_Alltoall(sendbuf, count, par::Mpi_datatype<T>::value(),
            recvbuf, count, par::Mpi_datatype<T>::value(), comm);

      PROF_PAR_ALL2ALL_END
    }

  template <typename T>
    inline int Mpi_Gather( T* sendBuffer, T* recvBuffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_GATHER_BEGIN

        MPI_Gather(sendBuffer, count, par::Mpi_datatype<T>::value(),
            recvBuffer, count, par::Mpi_datatype<T>::value(), root, comm);

      PROF_PAR_GATHER_END
    }

  template <typename T>
    inline int Mpi_Bcast(T* buffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_BCAST_BEGIN

        MPI_Bcast(buffer, count, par::Mpi_datatype<T>::value(), root, comm);

      PROF_PAR_BCAST_END
    }

  template <typename T>
    inline int Mpi_Reduce(T* sendbuf, T* recvbuf, int count, MPI_Op op, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_REDUCE_BEGIN

        MPI_Reduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, root, comm);

      PROF_PAR_REDUCE_END
    }

  template <typename T>
    int Mpi_Allgatherv(T* sendBuf, int sendCount, T* recvBuf, 
        int* recvCounts, int* displs, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALLGATHERV_BEGIN

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

        int maxSendCount;
      int npes, rank;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      par::Mpi_Allreduce<int>(&sendCount, &maxSendCount, 1, MPI_MAX, comm);

      T* dummySendBuf = new T[maxSendCount*npes];
      assert(dummySendBuf);

      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < sendCount; j++) {
          dummySendBuf[(i*maxSendCount) + j] = sendBuf[j];
        }
      }

      T* dummyRecvBuf = new T[maxSendCount*npes];
      assert(dummyRecvBuf);

      par::Mpi_Alltoall<T>(dummySendBuf, dummyRecvBuf, maxSendCount, comm);

      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < recvCounts[i]; j++) {
          recvBuf[displs[i] + j] = dummyRecvBuf[(i*maxSendCount) + j];
        }
      }

      delete [] dummySendBuf;
      delete [] dummyRecvBuf;

#else

      MPI_Allgatherv(sendBuf, sendCount, par::Mpi_datatype<T>::value(),
          recvBuf, recvCounts, displs, par::Mpi_datatype<T>::value(), comm);

#endif

      PROF_PAR_ALLGATHERV_END
    }

  template <typename T>
    int Mpi_Allgather(T* sendBuf, T* recvBuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALLGATHER_BEGIN

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

        int npes;
      MPI_Comm_size(comm, &npes);
      T* dummySendBuf = new T[count*npes];
      assert(dummySendBuf);
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < count; j++) {
          dummySendBuf[(i*count) + j] = sendBuf[j];
        }
      }
      par::Mpi_Alltoall<T>(dummySendBuf, recvBuf, count, comm);
      delete [] dummySendBuf;

#else

      MPI_Allgather(sendBuf, count, par::Mpi_datatype<T>::value(), 
          recvBuf, count, par::Mpi_datatype<T>::value(), comm);

#endif

      PROF_PAR_ALLGATHER_END
    }

  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALLV_SPARSE_BEGIN

        int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      int commCnt = 0;

      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      for(int i = (rank+1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      MPI_Request* requests = new MPI_Request[commCnt];
      assert(requests);

      MPI_Status* statuses = new MPI_Status[commCnt];
      assert(statuses);

      commCnt = 0;

      //First place all recv requests. Do not recv from self.
      for(int i = 0; i < rank; i++) {
        if(recvcnts[i] > 0) {
          par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(recvcnts[i] > 0) {
          par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Next send the messages. Do not send to self.
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], 
              i, 1, comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Now copy local portion.
#ifdef __DEBUG_PAR__
      assert(sendcnts[rank] == recvcnts[rank]);
#endif

      for(int i = 0; i < sendcnts[rank]; i++) {
        recvbuf[rdispls[rank] + i] = sendbuf[sdispls[rank] + i];
      }

      PROF_A2AV_WAIT_BEGIN

        MPI_Waitall(commCnt, requests, statuses);

      PROF_A2AV_WAIT_END

        delete [] requests;
      delete [] statuses;

      PROF_PAR_ALL2ALLV_SPARSE_END
    }

  template <typename T>
    int Mpi_Alltoallv_dense(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALLV_DENSE_BEGIN

        int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      //Processors may send a lot of information to themselves and a lesser
      //amount to others. If so, we don't want to waste communication by
      //including the local copy size in the max message size. 
      int maxNumElemSend = 0;
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > maxNumElemSend) {
          maxNumElemSend = sendcnts[i];
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > maxNumElemSend) {
          maxNumElemSend = sendcnts[i];
        }
      }

      int allToAllCount;
      par::Mpi_Allreduce<int>(&maxNumElemSend, &allToAllCount, 1, MPI_MAX, comm);

      T* tmpSendBuf = new T[allToAllCount*npes];
      assert(tmpSendBuf);

      T* tmpRecvBuf = new T[allToAllCount*npes];
      assert(tmpRecvBuf);

      for(int i = 0; i < rank; i++) {
        for(int j = 0; j < sendcnts[i]; j++) {
          tmpSendBuf[(allToAllCount*i) + j] = sendbuf[sdispls[i] + j];            
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        for(int j = 0; j < sendcnts[i]; j++) {
          tmpSendBuf[(allToAllCount*i) + j] = sendbuf[sdispls[i] + j];            
        }
      }

      par::Mpi_Alltoall<T>(tmpSendBuf, tmpRecvBuf, allToAllCount, comm);

      for(int i = 0; i < rank; i++) {
        for(int j = 0; j < recvcnts[i]; j++) {
          recvbuf[rdispls[i] + j] = tmpRecvBuf[(allToAllCount*i) + j];      
        }
      }

      //Now copy local portion.
#ifdef __DEBUG_PAR__
      assert(sendcnts[rank] == recvcnts[rank]);
#endif

      for(int j = 0; j < recvcnts[rank]; j++) {
        recvbuf[rdispls[rank] + j] = sendbuf[sdispls[rank] + j];      
      }

      for(int i = (rank + 1); i < npes; i++) {
        for(int j = 0; j < recvcnts[i]; j++) {
          recvbuf[rdispls[i] + j] = tmpRecvBuf[(allToAllCount*i) + j];      
        }
      }

      delete [] tmpSendBuf;
      delete [] tmpRecvBuf;

      PROF_PAR_ALL2ALLV_DENSE_END
    }

  template<typename T>
    unsigned int defaultWeight(const T *a){
      return 1;
    }

  template <typename T> 
    int scatterValues(std::vector<T> & in, std::vector<T> & out, 
        DendroIntL outSz, MPI_Comm comm ) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_SCATTER_BEGIN

        int rank, npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      MPI_Request request;
      MPI_Status status;

      DendroIntL inSz = in.size();
      out.resize(outSz);

      DendroIntL off1 = 0, off2 = 0;
      DendroIntL * scnIn = NULL;
      if(inSz) {  
        scnIn = new DendroIntL [inSz]; 
        assert(scnIn);
      }

      // perform a local scan first ...
      DendroIntL zero = 0;
      if(inSz) {
        scnIn[0] = 1;
        for (DendroIntL i = 1; i < inSz; i++) {
          scnIn[i] = scnIn[i-1] + 1;
        }//end for
        // now scan with the final members of 
        par::Mpi_Scan<DendroIntL>(scnIn+inSz-1, &off1, 1, MPI_SUM, comm ); 
      } else{
        par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm ); 
      }

      // communicate the offsets ...
      if (rank < (npes-1)){
        par::Mpi_Issend<DendroIntL>( &off1, 1, (rank + 1), 0, comm, &request );
      }
      if (rank){
        par::Mpi_Recv<DendroIntL>( &off2, 1, (rank - 1), 0, comm, &status );
      } else{
        off2 = 0; 
      }

      // add offset to local array
      for (DendroIntL i = 0; i < inSz; i++) {
        scnIn[i] = scnIn[i] + off2;  // This has the global scan results now ...
      }//end for

      //Gather Scan of outCnts
      DendroIntL *outCnts;
      outCnts = new DendroIntL[npes];
      assert(outCnts);

      if(rank < (npes-1)) {
        MPI_Status statusWait;
        MPI_Wait(&request,&statusWait);
      }

      if( outSz ) {
        par::Mpi_Scan<DendroIntL>( &outSz, &off1, 1, MPI_SUM, comm ); 
      }else {
        par::Mpi_Scan<DendroIntL>( &zero, &off1, 1, MPI_SUM, comm ); 
      }

      par::Mpi_Allgather<DendroIntL>( &off1, outCnts, 1, comm);

      int * sendSz = new int [npes];
      assert(sendSz);

      int * recvSz = new int [npes];
      assert(recvSz);

      int * sendOff = new int [npes];
      assert(sendOff);

      int * recvOff = new int [npes];
      assert(recvOff);

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...
      for (int i = 0; i < npes; i++) {
        sendSz[i] = 0;
      }

      //The Heart of the algorithm....
      //scnIn and outCnts are both sorted 
      DendroIntL inCnt = 0;
      int pCnt = 0;
      while( (inCnt < inSz) && (pCnt < npes) ) {
        if( scnIn[inCnt] <= outCnts[pCnt]  ) {
          sendSz[pCnt]++;
          inCnt++;
        }else {
          pCnt++;
        }
      }

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

      int nn=0; // new value of nlSize, ie the local nodes.
      for (int i=0; i<npes; i++) {
        nn += recvSz[i];
      }

      // compute offsets ...
      sendOff[0] = 0;
      recvOff[0] = 0;
      for (int i=1; i<npes; i++) {
        sendOff[i] = sendOff[i-1] + sendSz[i-1];
        recvOff[i] = recvOff[i-1] + recvSz[i-1];
      }

      assert(static_cast<unsigned int>(nn) == outSz);
      // perform All2All  ... 
      T* inPtr = NULL;
      T* outPtr = NULL;
      if(!in.empty()) {
        inPtr = &(*(in.begin()));
      }
      if(!out.empty()) {
        outPtr = &(*(out.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(inPtr, sendSz, sendOff, 
          outPtr, recvSz, recvOff, comm);

      // clean up...
      if(scnIn) {
        delete [] scnIn;
        scnIn = NULL;
      }

      delete [] outCnts;
      outCnts = NULL;

      delete [] sendSz;
      sendSz = NULL;

      delete [] sendOff;
      sendOff = NULL;

      delete [] recvSz;
      recvSz = NULL;

      delete [] recvOff;
      recvOff = NULL;

      PROF_PAR_SCATTER_END
    }


  template<typename T>
    int concatenate(std::vector<T> & listA, std::vector<T> & listB,
        MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_CONCAT_BEGIN

        int rank;
      int npes;

      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &npes);

      assert(!(listA.empty()));

      //1. First perform Allreduce to get total listA size
      //and total listB size; 

      DendroIntL locAsz_locBsz[2];
      DendroIntL globAsz_globBsz[2];

      locAsz_locBsz[0] = listA.size();
      locAsz_locBsz[1] = listB.size();
      globAsz_globBsz[0] = 0;
      globAsz_globBsz[1] = 0;

      par::Mpi_Allreduce<DendroIntL>(locAsz_locBsz, globAsz_globBsz, 2, MPI_SUM, comm);

      //2. Re-distribute A and B independently so that
      //B is distributed only on the high rank processors
      //and A is distribute only on the low rank processors.

      DendroIntL avgTotalSize = ((globAsz_globBsz[0] + globAsz_globBsz[1])/npes);

      //since listA is not empty on any of the active procs,
      //globASz > npes so avgTotalSize >= 1

      DendroIntL remTotalSize = ((globAsz_globBsz[0] + globAsz_globBsz[1])%npes);

      int numSmallProcs = (npes - remTotalSize);

      //In the final merged list, there will be exactly remTotalSize number
      //of processors each having (avgTotalSize + 1) elements and there will
      //be exactly numSmallProcs number of processors each having
      //avgTotalSize elements. 
      //Also, len(A) + len(B) = (numSmallProcs*avg) + (remTotalSize*(avg+1))

      std::vector<T> tmpA;
      std::vector<T> tmpB;

      int numAhighProcs;
      int numAlowProcs;
      int numBothProcs;
      int numBhighProcs;
      int numBlowProcs;
      DendroIntL aSizeForBoth;
      DendroIntL bSizeForBoth;

      if( globAsz_globBsz[1] <= (numSmallProcs*avgTotalSize) ) {
        numBhighProcs = 0;
        numBlowProcs = ((globAsz_globBsz[1])/avgTotalSize); 
        bSizeForBoth = ((globAsz_globBsz[1])%avgTotalSize);

        assert(numBlowProcs <= numSmallProcs);

        //remBsize is < avgTotalSize. So it will fit on one proc.
        if(bSizeForBoth) {
          numBothProcs = 1;
          if(numBlowProcs < numSmallProcs) {
            //We don't know if remTotalSize is 0 or not. 
            //So, let the common proc be a low proc.
            aSizeForBoth = (avgTotalSize - bSizeForBoth);
            numAhighProcs = remTotalSize;
            numAlowProcs = (numSmallProcs - (1 + numBlowProcs));
          } else {             
            //No more room for small procs. The common has to be a high proc.
            aSizeForBoth = ((avgTotalSize + 1) - bSizeForBoth);
            numAhighProcs = (remTotalSize - 1);
            numAlowProcs = 0;
          }
        } else {
          numBothProcs = 0;
          aSizeForBoth = 0;
          numAhighProcs = remTotalSize;
          numAlowProcs = (numSmallProcs - numBlowProcs);
        }
      } else {
        //Some B procs will have (avgTotalSize+1) elements
        DendroIntL numBusingAvgPlus1 = ((globAsz_globBsz[1])/(avgTotalSize + 1));
        DendroIntL remBusingAvgPlus1 = ((globAsz_globBsz[1])%(avgTotalSize + 1));
        if (numBusingAvgPlus1 <= remTotalSize) {
          //Each block can use (avg+1) elements each, since there will be some
          //remaining for A  
          numBhighProcs = numBusingAvgPlus1;
          numBlowProcs = 0;
          bSizeForBoth = remBusingAvgPlus1;
          if(bSizeForBoth) {
            numBothProcs = 1;
            if (numBhighProcs < remTotalSize) {
              //We don't know if numSmallProcs is 0 or not.
              //So, let the common proc be a high proc 
              aSizeForBoth = ((avgTotalSize + 1) - bSizeForBoth);
              numAhighProcs = (remTotalSize - (numBhighProcs + 1));
              numAlowProcs = numSmallProcs;
            } else {
              //No more room for high procs. The common has to be a low proc. 
              aSizeForBoth = (avgTotalSize - bSizeForBoth);
              numAhighProcs = 0;
              numAlowProcs = (numSmallProcs - 1);
            }
          } else {
            numBothProcs = 0;
            aSizeForBoth = 0;
            numAhighProcs = (remTotalSize - numBhighProcs);
            numAlowProcs = numSmallProcs;
          }
        } else {
          //Since numBusingAvgPlus1 > remTotalSize*(avg+1) 
          //=> len(B) > remTotalSize*(avg+1)
          //=> len(A) < numSmallProcs*avg
          //This is identical to the first case (except for 
          //the equality), with A and B swapped.

          assert( globAsz_globBsz[0] < (numSmallProcs*avgTotalSize) );

          numAhighProcs = 0;
          numAlowProcs = ((globAsz_globBsz[0])/avgTotalSize); 
          aSizeForBoth = ((globAsz_globBsz[0])%avgTotalSize);

          assert(numAlowProcs < numSmallProcs);

          //remAsize is < avgTotalSize. So it will fit on one proc.
          if(aSizeForBoth) {
            numBothProcs = 1;
            //We don't know if remTotalSize is 0 or not. 
            //So, let the common proc be a low proc.
            bSizeForBoth = (avgTotalSize - aSizeForBoth);
            numBhighProcs = remTotalSize;
            numBlowProcs = (numSmallProcs - (1 + numAlowProcs));
          } else {
            numBothProcs = 0;
            bSizeForBoth = 0;
            numBhighProcs = remTotalSize;
            numBlowProcs = (numSmallProcs - numAlowProcs);
          }
        }
      }

      assert((numAhighProcs + numAlowProcs + numBothProcs
            + numBhighProcs + numBlowProcs) == npes);

      assert((aSizeForBoth + bSizeForBoth) <= (avgTotalSize+1));

      if(numBothProcs) {
        assert((aSizeForBoth + bSizeForBoth) >= avgTotalSize);
      } else {
        assert(aSizeForBoth == 0); 
        assert(bSizeForBoth == 0); 
      }

      if((aSizeForBoth + bSizeForBoth) == (avgTotalSize + 1)) {
        assert((numAhighProcs + numBothProcs + numBhighProcs) == remTotalSize);
        assert((numAlowProcs + numBlowProcs) == numSmallProcs);
      } else {
        assert((numAhighProcs + numBhighProcs) == remTotalSize);
        assert((numAlowProcs + numBothProcs + numBlowProcs) == numSmallProcs);
      }

      //The partition is as follow:
      //1. numAhighProcs with (avg+1) elements each exclusively from A,
      //2. numAlowProcs with avg elements each exclusively from A
      //3. numBothProcs with aSizeForBoth elements from A and
      // bSizeForBoth elements from B
      //4. numBhighProcs with (avg+1) elements each exclusively from B.
      //5. numBlowProcs with avg elements each exclusively from B.

      if(rank < numAhighProcs) {
        par::scatterValues<T>(listA, tmpA, (avgTotalSize + 1), comm);
        par::scatterValues<T>(listB, tmpB, 0, comm);
      } else if (rank < (numAhighProcs + numAlowProcs)) {
        par::scatterValues<T>(listA, tmpA, avgTotalSize, comm);
        par::scatterValues<T>(listB, tmpB, 0, comm);
      } else if (rank < (numAhighProcs + numAlowProcs + numBothProcs)) {
        par::scatterValues<T>(listA, tmpA, aSizeForBoth, comm);
        par::scatterValues<T>(listB, tmpB, bSizeForBoth, comm);
      } else if (rank <
          (numAhighProcs + numAlowProcs + numBothProcs + numBhighProcs)) {
        par::scatterValues<T>(listA, tmpA, 0, comm);
        par::scatterValues<T>(listB, tmpB, (avgTotalSize + 1), comm);
      } else {
        par::scatterValues<T>(listA, tmpA, 0, comm);
        par::scatterValues<T>(listB, tmpB, avgTotalSize, comm);
      }

      listA = tmpA;
      listB = tmpB;
      tmpA.clear();
      tmpB.clear();

      //3. Finally do a simple concatenation A = A + B. If the previous step
      //was performed correctly, there will be atmost 1 processor, which has both
      //non-empty A and non-empty B. On other processors one of the two lists
      //will be empty
      if(listA.empty()) {
        listA = listB;
      } else {
        if(!(listB.empty())) {
          listA.insert(listA.end(), listB.begin(), listB.end());
        }
      }

      listB.clear();

      PROF_PAR_CONCAT_END
    }

  template <typename T>
    int maxLowerBound(const std::vector<T> & keys, const std::vector<T> & searchList,
        std::vector<T> & results, MPI_Comm comm) {
      PROF_SEARCH_BEGIN

        int rank, npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      // allocate memory for the mins array
      std::vector<T> mins (npes);
      assert(!searchList.empty());

      T* searchListPtr = NULL;
      T* minsPtr = NULL;
      if(!searchList.empty()) {
        searchListPtr = &(*(searchList.begin()));
      }
      if(!mins.empty()) {
        minsPtr = &(*(mins.begin()));
      }
      par::Mpi_Allgather<T>(searchListPtr, minsPtr, 1, comm);

      //For each key decide which processor to send to
      unsigned int *part = NULL;

      if(keys.size()) {
        part = new unsigned int[keys.size()];
        assert(part);
      }

      for ( unsigned int i=0; i<keys.size(); i++ ) {
        //maxLB returns the smallest index in a sorted array such
        //that a[ind] <= key and  a[index +1] > key
        bool found = par::maxLowerBound<T>(mins,keys[i], part+i,NULL,NULL);
        if ( !found ) {
          //This key is smaller than the mins from every processor.
          //No point in searching.
          part[i] = rank;
        }
      }

      mins.clear();

      int *numKeysSend = new int[npes];
      assert(numKeysSend);

      int *numKeysRecv = new int[npes];
      assert(numKeysRecv);

      for ( int i=0; i<npes; i++ ) {
        numKeysSend[i] = 0;
      }

      // calculate the number of keys to send ...
      for ( unsigned int i=0; i<keys.size(); i++ ) {
        numKeysSend[part[i]]++;
      }

      // Now do an All2All to get numKeysRecv
      par::Mpi_Alltoall<int>(numKeysSend, numKeysRecv, 1, comm);

      unsigned int totalKeys=0;	// total number of local keys ...
      for ( int i=0; i<npes; i++ ) {
        totalKeys += numKeysRecv[i];
      }

      // create the send and recv buffers ...
      std::vector<T> sendK (keys.size());
      std::vector<T> recvK (totalKeys);

      // the mapping ..
      unsigned int * comm_map = NULL;

      if(keys.size()) {
        comm_map = new unsigned int [keys.size()];
        assert(comm_map);
      }

      // Now create sendK
      int *sendOffsets = new int[npes]; 
      assert(sendOffsets);
      sendOffsets[0] = 0;

      int *recvOffsets = new int[npes]; 
      assert(recvOffsets);
      recvOffsets[0] = 0;

      int *numKeysTmp = new int[npes]; 
      assert(numKeysTmp);
      numKeysTmp[0] = 0; 

      // compute offsets ...
      for ( int i=1; i<npes; i++ ) {
        sendOffsets[i] = sendOffsets[i-1] + numKeysSend[i-1];
        recvOffsets[i] = recvOffsets[i-1] + numKeysRecv[i-1];
        numKeysTmp[i] = 0; 
      }

      for ( unsigned int i=0; i< keys.size(); i++ ) {
        unsigned int ni = numKeysTmp[part[i]];
        numKeysTmp[part[i]]++;
        // set entry ...
        sendK[sendOffsets[part[i]] + ni] = keys[i];
        // save mapping .. will need it later ...
        comm_map[i] = sendOffsets[part[i]] + ni;
      }

      if(part) {
        delete [] part;
      }

      assert(numKeysTmp);
      delete [] numKeysTmp;
      numKeysTmp = NULL;

      T* sendKptr = NULL;
      T* recvKptr = NULL;
      if(!sendK.empty()) {
        sendKptr = &(*(sendK.begin()));
      }
      if(!recvK.empty()) {
        recvKptr = &(*(recvK.begin()));
      }

      par::Mpi_Alltoallv_sparse<T>(sendKptr, numKeysSend, sendOffsets, 
          recvKptr, numKeysRecv, recvOffsets, comm);


      std::vector<T>  resSend (totalKeys);
      std::vector<T>  resRecv (keys.size());

      //Final local search.
      for ( unsigned int i = 0; i < totalKeys; i++) {
        unsigned int idx;
        bool found = par::maxLowerBound<T>( searchList, recvK[i], &idx,NULL,NULL );
        if(found) {
          resSend[i] = searchList[idx];
        }
      }//end for i

      //Exchange Results
      //Return what you received in the earlier communication.
      T* resSendPtr = NULL;
      T* resRecvPtr = NULL;
      if(!resSend.empty()) {
        resSendPtr = &(*(resSend.begin()));
      }
      if(!resRecv.empty()) {
        resRecvPtr = &(*(resRecv.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(resSendPtr, numKeysRecv, recvOffsets, 
          resRecvPtr, numKeysSend, sendOffsets, comm);

      assert(sendOffsets);
      delete [] sendOffsets;
      sendOffsets = NULL;

      assert(recvOffsets);
      delete [] recvOffsets;
      recvOffsets = NULL;

      assert(numKeysSend);
      delete [] numKeysSend;
      numKeysSend = NULL;

      assert(numKeysRecv);
      delete [] numKeysRecv;
      numKeysRecv = NULL;

      for ( unsigned int i=0; i < keys.size(); i++ ) {
        results[i] = resRecv[comm_map[i]];  
      }//end for

      // Clean up ...
      if(comm_map) {
        delete [] comm_map;
      }

      PROF_SEARCH_END
    }

  template<typename T>
    int partitionW(std::vector<T>& nodeList, unsigned int (*getWeight)(const T *), MPI_Comm comm){
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PARTW_BEGIN

        int npes;

      MPI_Comm_size(comm, &npes);

      if(npes == 1) {
        PROF_PARTW_END
      }

      if(getWeight == NULL) {
        getWeight = par::defaultWeight<T>;
      }

      int rank;

      MPI_Comm_rank(comm, &rank);

      MPI_Request request;
      MPI_Status status;
      const bool nEmpty = nodeList.empty();

      DendroIntL  off1= 0, off2= 0, localWt= 0, totalWt = 0;

      DendroIntL* wts = NULL;
      DendroIntL* lscn = NULL;
      DendroIntL nlSize = nodeList.size();
      if(nlSize) {
        wts = new DendroIntL[nlSize];
        assert(wts);

        lscn= new DendroIntL[nlSize]; 
        assert(lscn);
      }

      // First construct arrays of id and wts.
      for (DendroIntL i = 0; i < nlSize; i++) {
        wts[i] = (*getWeight)( &(nodeList[i]) );
        localWt += wts[i];
      }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-1 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // compute the total weight of the problem ...
      par::Mpi_Allreduce<DendroIntL>(&localWt, &totalWt, 1, MPI_SUM, comm);

      // perform a local scan on the weights first ...
      DendroIntL zero = 0;
      if(!nEmpty) {
        lscn[0]=wts[0];
        for (DendroIntL i = 1; i < nlSize; i++) {
          lscn[i] = wts[i] + lscn[i-1];
        }//end for
        // now scan with the final members of 
        par::Mpi_Scan<DendroIntL>(lscn+nlSize-1, &off1, 1, MPI_SUM, comm ); 
      } else{
        par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm ); 
      }

      // communicate the offsets ...
      if (rank < (npes-1)){
        par::Mpi_Issend<DendroIntL>( &off1, 1, rank+1, 0, comm, &request );
      }
      if (rank){
        par::Mpi_Recv<DendroIntL>( &off2, 1, rank-1, 0, comm, &status );
      }
      else{
        off2 = 0; 
      }

      // add offset to local array
      for (DendroIntL i = 0; i < nlSize; i++) {
        lscn[i] = lscn[i] + off2;       // This has the global scan results now ...
      }//end for

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-2 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      int * sendSz = new int [npes];
      assert(sendSz);

      int * recvSz = new int [npes];
      assert(recvSz);

      int * sendOff = new int [npes]; 
      assert(sendOff);
      sendOff[0] = 0;

      int * recvOff = new int [npes]; 
      assert(recvOff);
      recvOff[0] = 0;

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...

      for (int i = 0; i < npes; i++) {
        sendSz[i] = 0;
      }

      // Now determine the average load ...
      DendroIntL npesLong = npes;
      DendroIntL avgLoad = (totalWt/npesLong);

      DendroIntL extra = (totalWt%npesLong);

      //The Heart of the algorithm....
      if(avgLoad > 0) {
        for (DendroIntL i = 0; i < nlSize; i++) {
          if(lscn[i] == 0) {		
            sendSz[0]++;
          }else {
            int ind=0;
            if ( lscn[i] <= (extra*(avgLoad + 1)) ) {
              ind = ((lscn[i] - 1)/(avgLoad + 1));
            }else {
              ind = ((lscn[i] - (1 + extra))/avgLoad);
            }
            assert(ind < npes);
            sendSz[ind]++;
          }//end if-else
        }//end for 
      }else {
        sendSz[0]+= nlSize;
      }//end if-else

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-3 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      if(rank < (npes-1)) {
        MPI_Status statusWait;
        MPI_Wait(&request, &statusWait);
      }

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

#ifdef __DEBUG_PAR__
      DendroIntL totSendToOthers = 0;
      DendroIntL totRecvFromOthers = 0;
      for (int i = 0; i < npes; i++) {
        if(rank != i) {
          totSendToOthers += sendSz[i];
          totRecvFromOthers += recvSz[i];
        }
      }
#endif

      DendroIntL nn=0; // new value of nlSize, ie the local nodes.
      for (int i = 0; i < npes; i++) {
        nn += recvSz[i];
      }

      // compute offsets ...
      for (int i = 1; i < npes; i++) {
        sendOff[i] = sendOff[i-1] + sendSz[i-1];
        recvOff[i] = recvOff[i-1] + recvSz[i-1];
      }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-4 passed."<<std::endl;
      }
      MPI_Barrier(comm);
      /*
         std::cout<<rank<<": newSize: "<<nn<<" oldSize: "<<(nodeList.size())
         <<" send: "<<totSendToOthers<<" recv: "<<totRecvFromOthers<<std::endl;
       */
      MPI_Barrier(comm);
#endif

      // allocate memory for the new arrays ...
      std::vector<T > newNodes(nn);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Final alloc successful."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // perform All2All  ... 
      T* nodeListPtr = NULL;
      T* newNodesPtr = NULL;
      if(!nodeList.empty()) {
        nodeListPtr = &(*(nodeList.begin()));
      }
      if(!newNodes.empty()) {
        newNodesPtr = &(*(newNodes.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(nodeListPtr, sendSz, sendOff, 
          newNodesPtr, recvSz, recvOff, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-5 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // reset the pointer ...
      nodeList = newNodes;
      newNodes.clear();

      // clean up...
      if(!nEmpty) {
        delete [] lscn;
        delete [] wts;
      }
      delete [] sendSz;
      sendSz = NULL;

      delete [] sendOff;
      sendOff = NULL;

      delete [] recvSz;
      recvSz = NULL;

      delete [] recvOff;
      recvOff = NULL;

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-6 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      PROF_PARTW_END
    }//end function

  template<typename T>
    int removeDuplicates(std::vector<T>& vecT, bool isSorted, MPI_Comm comm){
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_REMDUP_BEGIN
        int size, rank;
      MPI_Comm_size(comm,&size);
      MPI_Comm_rank(comm,&rank);

      std::vector<T> tmpVec;
      if(!isSorted) {	  	
        //Sort partitions vecT and tmpVec internally.
        par::sampleSort<T>(vecT, tmpVec, comm);	  		  	
      }else {
        tmpVec = vecT;
      }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-1 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      vecT.clear();
      par::partitionW<T>(tmpVec, NULL, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-2 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      //Remove duplicates locally
      seq::makeVectorUnique<T>(tmpVec,true); 

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-3 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      //Creating groups

      int new_rank, new_size; 
      MPI_Comm   new_comm;
      par::splitComm2way(tmpVec.empty(), &new_comm, comm);

      MPI_Comm_rank (new_comm, &new_rank);
      MPI_Comm_size (new_comm, &new_size);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-4 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      //Checking boundaries... 
      if(!tmpVec.empty()) {
        T end = tmpVec[tmpVec.size()-1];	  
        T endRecv;

        //communicate end to the next processor.
        MPI_Status status;

        par::Mpi_Sendrecv<T, T>(&end, 1, ((new_rank <(new_size-1))?(new_rank+1):0), 1, &endRecv,
            1, ((new_rank > 0)?(new_rank-1):(new_size-1)), 1, new_comm, &status);

        //Remove endRecv if it exists (There can be no more than one copy of this)
        if(new_rank) {
          typename std::vector<T>::iterator Iter = find(tmpVec.begin(),tmpVec.end(),endRecv);
          if(Iter != tmpVec.end()) {
            tmpVec.erase(Iter);
          }//end if found    
        }//end if p not 0	  
      }//end if not empty

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-5 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      vecT = tmpVec;
      tmpVec.clear();
      par::partitionW<T>(vecT, NULL, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-6 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      PROF_REMDUP_END
    }//end function

  template<typename T>
    int sampleSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm){ 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_SORT_BEGIN

        int npes;

      MPI_Comm_size(comm, &npes);

      if (npes == 1) {
        std::cout <<" have to use seq. sort"
          <<" since npes = 1 . inpSize: "<<(arr.size()) <<std::endl;
        std::sort(arr.begin(), arr.end());
        SortedElem  = arr;
        PROF_SORT_END
      } 

      std::vector<T>  splitters;
      std::vector<T>  allsplitters;

      int myrank;
      MPI_Comm_rank(comm, &myrank);

      DendroIntL nelem = arr.size();
      DendroIntL nelemCopy = nelem;
      DendroIntL totSize;
      par::Mpi_Allreduce<DendroIntL>(&nelemCopy, &totSize, 1, MPI_SUM, comm);

      DendroIntL npesLong = npes;
      const DendroIntL FIVE = 5;

      if(totSize < (FIVE*npesLong*npesLong)) {
        if(!myrank) {
          std::cout <<" Using bitonic sort since totSize < (5*(npes^2)). totSize: "
            <<totSize<<" npes: "<<npes <<std::endl;
        }
        par::partitionW<T>(arr, NULL, comm);

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-1 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        SortedElem = arr; 
        MPI_Comm new_comm;
        if(totSize < npesLong) {
          if(!myrank) {
            std::cout<<" Input to sort is small. splittingComm: "
              <<npes<<" -> "<< totSize<<std::endl;
          }
          par::splitCommUsingSplittingRank(static_cast<int>(totSize), &new_comm, comm);
        } else {
          new_comm = comm;
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-2 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        if(!SortedElem.empty()) {
          par::bitonicSort<T>(SortedElem, new_comm);
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-3 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        PROF_SORT_END
      }// end if

#ifdef __DEBUG_PAR__
      if(!myrank) {
        std::cout<<"Using sample sort to sort nodes. n/p^2 is fine."<<std::endl;
      }
#endif

      //Re-part arr so that each proc. has atleast p elements.
      par::partitionW<T>(arr, NULL, comm);

      nelem = arr.size();

      std::sort(arr.begin(),arr.end());

      std::vector<T> sendSplits(npes-1);
      splitters.resize(npes);

      for(int i = 1; i < npes; i++)	 {
        sendSplits[i-1] = arr[i*nelem/npes];	
      }//end for i

      // sort sendSplits using bitonic ...
      par::bitonicSort<T>(sendSplits,comm);

      // All gather with last element of splitters.
      T* sendSplitsPtr = NULL;
      T* splittersPtr = NULL;
      if(sendSplits.size() > static_cast<unsigned int>(npes-2)) {
        sendSplitsPtr = &(*(sendSplits.begin() + (npes -2)));
      }
      if(!splitters.empty()) {
        splittersPtr = &(*(splitters.begin()));
      }
      par::Mpi_Allgather<T>(sendSplitsPtr, splittersPtr, 1, comm);

      sendSplits.clear();

      int *sendcnts = new int[npes];
      assert(sendcnts);

      int * recvcnts = new int[npes];
      assert(recvcnts);

      int * sdispls = new int[npes];
      assert(sdispls);

      int * rdispls = new int[npes];
      assert(rdispls);

      for(int k = 0; k < npes; k++){
        sendcnts[k] = 0;
      }

      int k = 0;

      for (DendroIntL j = 0; j < nelem; j++) {
        if (arr[j] <= splitters[k]) {
          sendcnts[k]++;
        } else{
          k = seq::UpperBound<T>(npes-1, splittersPtr, k+1, arr[j]);
          if (k == (npes-1) ){
            //could not find any splitter >= arr[j]
            sendcnts[k] = (nelem - j);
            break;
          } else {
            assert(k < (npes-1));
            assert(splitters[k] >= arr[j]);
            sendcnts[k]++;
          }
        }//end if-else
      }//end for j

      par::Mpi_Alltoall<int>(sendcnts, recvcnts, 1, comm);

      sdispls[0] = 0; rdispls[0] = 0;
      for (int j = 1; j < npes; j++){
        sdispls[j] = sdispls[j-1] + sendcnts[j-1];
        rdispls[j] = rdispls[j-1] + recvcnts[j-1];
      }

      DendroIntL nsorted = rdispls[npes-1] + recvcnts[npes-1];
      SortedElem.resize(nsorted);

      T* arrPtr = NULL;
      T* SortedElemPtr = NULL;
      if(!arr.empty()) {
        arrPtr = &(*(arr.begin()));
      }
      if(!SortedElem.empty()) {
        SortedElemPtr = &(*(SortedElem.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(arrPtr, sendcnts, sdispls,
          SortedElemPtr, recvcnts, rdispls, comm);

      arr.clear();

      delete [] sendcnts;
      sendcnts = NULL;

      delete [] recvcnts;
      recvcnts = NULL;

      delete [] sdispls;
      sdispls = NULL;

      delete [] rdispls;
      rdispls = NULL;

      sort(SortedElem.begin(), SortedElem.end());

      PROF_SORT_END
    }//end function

  /********************************************************************/
  /*
   * which_keys is one of KEEP_HIGH or KEEP_LOW
   * partner    is the processor with which to Merge and Split.
   *
   */
  template <typename T>
    void MergeSplit( std::vector<T> &local_list, int which_keys, int partner, MPI_Comm  comm) {

      MPI_Status status;
      int send_size = local_list.size();
      int recv_size = 0;

      // first communicate how many you will send and how many you will receive ...

      par::Mpi_Sendrecv<int, int>( &send_size , 1, partner, 0,
          &recv_size, 1, partner, 0, comm, &status);

      std::vector<T> temp_list( recv_size );

      T* local_listPtr = NULL;
      T* temp_listPtr = NULL;
      if(!local_list.empty()) {
        local_listPtr = &(*(local_list.begin()));
      }
      if(!temp_list.empty()) {
        temp_listPtr = &(*(temp_list.begin()));
      }

      par::Mpi_Sendrecv<T, T>( local_listPtr, send_size, partner,
          1, temp_listPtr, recv_size, partner, 1, comm, &status);

      MergeLists<T>(local_list, temp_list, which_keys);

      temp_list.clear();
    } // Merge_split 

  template <typename T>
    void Par_bitonic_sort_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm ) {
      int  eor_bit;
      int       proc_set_dim;
      int       stage;
      int       partner;
      int       my_rank;

      MPI_Comm_rank(comm, &my_rank);

      proc_set_dim = 0;
      int x = proc_set_size;
      while (x > 1) {
        x = x >> 1;
        proc_set_dim++;
      }

      eor_bit = (1 << (proc_set_dim - 1) );
      for (stage = 0; stage < proc_set_dim; stage++) {
        partner = (my_rank ^ eor_bit);

        if (my_rank < partner) {
          MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
        } else {
          MergeSplit<T> ( local_list, KEEP_HIGH, partner, comm);
        }

        eor_bit = (eor_bit >> 1);
      }
    }  // Par_bitonic_sort_incr 


  template <typename T>
    void Par_bitonic_sort_decr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm) {
      int  eor_bit;
      int       proc_set_dim;
      int       stage;
      int       partner;
      int       my_rank;

      MPI_Comm_rank(comm, &my_rank);

      proc_set_dim = 0;
      int x = proc_set_size;
      while (x > 1) {
        x = x >> 1;
        proc_set_dim++;
      }

      eor_bit = (1 << (proc_set_dim - 1));
      for (stage = 0; stage < proc_set_dim; stage++) {
        partner = my_rank ^ eor_bit;

        if (my_rank > partner) {
          MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
        } else {
          MergeSplit<T> ( local_list, KEEP_HIGH, partner, comm);
        }

        eor_bit = (eor_bit >> 1);
      }

    } // Par_bitonic_sort_decr 

  template <typename T>
    void Par_bitonic_merge_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm ) {
      int       partner;
      int       rank, npes;

      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &npes);

      unsigned int num_left  =  binOp::getPrevHighestPowerOfTwo(npes);
      unsigned int num_right = npes - num_left;

      // 1, Do merge between the k right procs and the highest k left procs.
      if ( (static_cast<unsigned int>(rank) < num_left) &&
          (static_cast<unsigned int>(rank) >= (num_left - num_right)) ) {
        partner = static_cast<unsigned int>(rank) + num_right;
        MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
      } else if (static_cast<unsigned int>(rank) >= num_left) {
        partner = static_cast<unsigned int>(rank) - num_right;
        MergeSplit<T> ( local_list,  KEEP_HIGH, partner, comm);
      }
    }

  template <typename T>
    void bitonicSort_binary(std::vector<T> & in, MPI_Comm comm) {
      int       	    proc_set_size;
      unsigned int	    and_bit;
      int       	rank;
      int       	npes;

      MPI_Comm_size(comm, &npes);

#ifdef __DEBUG_PAR__
      assert(npes > 1);
      assert(!(npes & (npes-1)));
      assert(!(in.empty()));
#endif

      MPI_Comm_rank(comm, &rank);

      for (proc_set_size = 2, and_bit = 2;
          proc_set_size <= npes;
          proc_set_size = proc_set_size*2, 
          and_bit = and_bit << 1) {

        if ((rank & and_bit) == 0) {
          Par_bitonic_sort_incr<T>( in, proc_set_size, comm);
        } else {
          Par_bitonic_sort_decr<T>( in, proc_set_size, comm);
        }
      }//end for
    }

  template <typename T>
    void bitonicSort(std::vector<T> & in, MPI_Comm comm) {
      int       	rank;
      int       	npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      assert(!(in.empty()));

      //Local Sort first
      std::sort(in.begin(),in.end());

      if(npes > 1) {

        // check if npes is a power of two ...
        bool isPower = (!(npes & (npes - 1)));

        if ( isPower ) {
          bitonicSort_binary<T>(in, comm);
        } else {
          MPI_Comm new_comm;

          // Since npes is not a power of two, we shall split the problem in two ...
          //
          // 1. Create 2 comm groups ... one for the 2^d portion and one for the
          // remainder.
          unsigned int splitter = splitCommBinary(comm, &new_comm);

          if ( static_cast<unsigned int>(rank) < splitter) {
            bitonicSort_binary<T>(in, new_comm);
          } else {
            bitonicSort<T>(in, new_comm);
          }

          // 3. Do a special merge of the two segments. (original comm).
          Par_bitonic_merge_incr( in,  binOp::getNextHighestPowerOfTwo(npes), comm );

          splitter = splitCommBinaryNoFlip(comm, &new_comm);

          // 4. Now a final sort on the segments.
          if (static_cast<unsigned int>(rank) < splitter) {
            bitonicSort_binary<T>(in, new_comm);
          } else {
            bitonicSort<T>(in, new_comm);
          }
        }//end if isPower of 2
      }//end if single processor
    }//end function

  template <typename T>
    void MergeLists( std::vector<T> &listA, std::vector<T> &listB,
        int KEEP_WHAT) {

      T _low, _high;

      assert(!(listA.empty()));
      assert(!(listB.empty()));

      _low  = ( (listA[0] > listB[0]) ? listA[0] : listB[0]);
      _high = ( (listA[listA.size()-1] < listB[listB.size()-1]) ?
          listA[listA.size()-1] : listB[listB.size()-1]);

      // We will do a full merge first ...
      unsigned int list_size = static_cast<unsigned int>(listA.size() + listB.size());

      std::vector<T> scratch_list(list_size);

      unsigned int  index1 = 0;
      unsigned int  index2 = 0; 

      for (int i = 0; i < list_size; i++) {
        //The order of (A || B) is important here, 
        //so that index2 remains within bounds
        if ( (index1 < listA.size()) && 
            ( (index2 >= listB.size()) ||
              (listA[index1] <= listB[index2]) ) ) {
          scratch_list[i] = listA[index1];
          index1++;
        } else {
          scratch_list[i] = listB[index2];
          index2++;	
        }
      }

      //Scratch list is sorted at this point.

      listA.clear();
      listB.clear();
      if ( KEEP_WHAT == KEEP_LOW ) {
        int ii=0;
        while ( ( (scratch_list[ii] < _low) ||
              (ii < (list_size/2)) )
            && (scratch_list[ii] <= _high) ) {
          ii++;	
        }
        if(ii) {
          listA.insert(listA.end(), scratch_list.begin(),
              (scratch_list.begin() + ii));
        }
      } else {
        int ii = (list_size - 1);
        while ( ( (ii >= (list_size/2)) 
              && (scratch_list[ii] >= _low) )
            || (scratch_list[ii] > _high) ) {
          ii--;	
        }
        if(ii < (list_size - 1) ) {
          listA.insert(listA.begin(), (scratch_list.begin() + (ii + 1)),
              (scratch_list.begin() + list_size));
        }
      }
      scratch_list.clear();
    }//end function

}//end namespace

