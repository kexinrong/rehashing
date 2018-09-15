#include "knnreduce.h"
#include <cfloat>

namespace knn {

        void query_r(MPI_Comm comm, double R, dist_t * arr, long nq, long len,
                        dist_t *& retarr, long *& retlen)
        {
		retlen = new long[nq];
		retarr = new dist_t[nq * len];
		int added = 0;
		for(long i = 0; i < nq; i++)
		{
			int localAdded = 0;
			for(long j = 0; j < len; j++)
			{
				if(arr[i * len + j].first <= R)
				{
					retarr[added] = arr[i * len + j];
					added++;
					localAdded++;
				}
			}
			retlen[i] = localAdded;
		}
        }

        long gnq;
        long gK;

        void op_merge(void * a, void * b, int * size, MPI_Datatype * type)
        {

/*
                //cout << "gnq " << gnq << " gk " << gK << " size = " << *size << " oursize = "<< gnq << endl;
                dist_t* inbuf = (dist_t*)a;
                dist_t* outbuf = (dist_t*)b;
//#pragma omp parallel for
                for(int i=0; i < *size; ++i) {
                        dist_t * start1 = inbuf + i*gK;
                        dist_t * end1 = start1 + gK;
                        dist_t * start2 = outbuf + i*gK;
                        dist_t * end2 = start2 + gK;
                        std::vector<dist_t> vin(start1, end1);
                        std::vector<dist_t> vout(start2, end2);
                        std::vector<dist_t> res(vin.size() + vout.size());
                        std::merge(vin.begin(), vin.end(), vout.begin(), vout.end(), res.begin());
                        for(int j=0; j < gK; ++j) {
                                start2[j] = res[j];
                        }
                }
*/

		pair<double,long>* A = (pair<double,long>*) a;
		pair<double,long>* B = (pair<double,long>*) b;

		int k = gK;
		vector< pair<double,long> > result((*size)*k);
	        #pragma omp parallel for
	        for(int i = 0; i < *size; i++) {
	                int aloc = i*k;
	                int bloc = i*k;
			int resultloc = i*k;
 	                for(int j = 0; j < k; j++) {
				if( (A[aloc].second == B[bloc].second) && (bloc == (i+1)*k-1) ) {
					B[bloc].first = DBL_MAX;
					B[bloc].second = -1;
					continue;
				}
				if( (A[aloc].second == B[bloc].second) && (bloc < (i+1)*k-1) ) bloc++;
                        	if( A[aloc] <= B[bloc] ) {
                                	result[resultloc++] = A[aloc++];
                        	} else {
                                	result[resultloc++] = B[bloc++];
                       		}
                	}
        	} // end for (i < n)

		#pragma omp parallel for
		for(int i = 0; i < (*size)*k; i ++)
			B[i] = result[i];

        }

        void query_k(MPI_Comm comm, int K, int root, dist_t * arr, int nquery, long nlocal, dist_t *& retarr)
        {
                gK = K;
                gnq = nquery;

                MPI_Op OP_MERGE;
                MPI_Op_create(op_merge, 1, &OP_MERGE);

                dist_t * inbuf = new dist_t[nquery * K];
                retarr = new dist_t[nquery * K];
		#pragma omp parallel for
                for(int i=0; i < nquery; ++i) {
                        for(int j=0; j < K; ++j) {
                                if(j < nlocal)
                                {
                                        inbuf[i*K + j] = arr[i*nlocal + j];
                                }
                                else
                                {
                                        inbuf[i*K + j].first = std::numeric_limits<double>::max();
                                        inbuf[i*K + j].second = std::numeric_limits<long>::max();
                                }
                        }
                }

                MPI_Datatype pairtype;
                MPI_Type_contiguous(K * sizeof(dist_t), MPI_BYTE, &pairtype);
                //MPI_Type_contiguous(K * sizeof(dist_t), MPI_CHAR, &pairtype);
                MPI_Type_commit(&pairtype);
                MPI_Reduce(inbuf, retarr, nquery, pairtype, OP_MERGE, root, comm);
                MPI_Op_free(&OP_MERGE);
                MPI_Type_free(&pairtype);
                delete[] inbuf;
        }
}
