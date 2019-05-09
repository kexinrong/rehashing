#ifndef __PAPI_PERF_H__
#define __PAPI_PERF_H__

  #include <mpi.h>
  #include <cassert>
  #include <pthread.h>
  //Only on systems with PAPI installed
  #ifdef PAPI 
    #include <papi.h>
    #include <omp.h>
    #include <cstdio>
  #endif


  void init_papi() {
  #ifdef PAPI 
    //Must call before PAPI_thread_init
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT){
      printf("%s:%d::PAPI_library_init failed. %d\n",
             __FILE__,__LINE__,retval);
      exit(1);
    }
     //Set up multi-threading in PAPI.  Call this only once.
     retval = PAPI_thread_init( (unsigned long (*)(void))(pthread_self) );
  #endif  
  }




  int papi_thread_flops( float *rtime, float *ptime, long long *flpops, float *mflops_total ) {
  #ifdef PAPI 
    int maxt = omp_get_max_threads();
    float real_time[maxt], proc_time[maxt], mflops[maxt];
    long long flpins[maxt];
    int retval = PAPI_OK;
    retval = PAPI_flops(&(real_time[maxt]), &(proc_time[maxt]), &(flpins[maxt]), &(mflops[maxt]));
 
    #pragma omp parallel 
    {
       int retval;
       int t = omp_get_thread_num();
       /* Collect the data into the variables passed in */
       retval = PAPI_flops(&(real_time[t]), &(proc_time[t]), &(flpins[t]), &(mflops[t]));
    }
 
    int t;
    *flpops = 0;
    *mflops_total = 0.0;
    *rtime = 0.0;
    *ptime = 0.0;
    //Combine all threads' results.
    for(t = 0; t < maxt+1; t++) {
       //Use the maximum time, but they should be practically equal.
       if(real_time[t] > *rtime) *rtime = real_time[t];
       if(proc_time[t] > *ptime) *ptime = proc_time[t];
       *flpops += flpins[t];
       //Overall MFLOPS/s is approximately the sum (assuming all threads have the same proc_time).
       *mflops_total += mflops[t];
    }

    return retval;
  #else
    return -1;
  #endif  
  }


  void papi_mpi_flop_start() {
      float global_real_time=0.0, global_proc_time=0.0, global_mflops=0.0;
      long long global_flpins = 0;
      #ifdef PAPI
      MPI_Barrier(MPI_COMM_WORLD);
         int retval;
         if (papi_thread_flops( &global_real_time, &global_proc_time, &global_flpins, &global_mflops) < PAPI_OK) {
            fprintf(stderr, "%s:%d ", __FILE__, __LINE__);
            fprintf(stderr, "PAPI_flips failed with return value = %d\n", retval);
            exit(1);
         }
      #endif
  }


  float papi_mpi_flop_stop() {
      float global_real_time=0.0, global_proc_time=0.0, global_mflops=0.0;
      long long global_flpins = 0;
      float total_global_mflops = -1.0;
      MPI_Barrier(MPI_COMM_WORLD);
      #ifdef PAPI
         int retval;
         if (papi_thread_flops( &global_real_time, &global_proc_time, &global_flpins, &global_mflops) < PAPI_OK) {
            fprintf(stderr, "%s:%d ", __FILE__, __LINE__);
            fprintf(stderr, "PAPI_flips failed with return value = %d\n", retval);
            exit(1);
         }
         MPI_Allreduce(&global_mflops, &total_global_mflops, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      #endif
      return total_global_mflops;
  }


 
#endif
