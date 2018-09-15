
#ifndef ASKIT_UTILS_IMPL_HPP_
#define ASKIT_UTILS_IMPL_HPP_

namespace askit {

  // uses kernel (and workspace in K), to compute K * q
  template<class TKernel>
  void ComputeKernelMatvec(TKernel& kernel, std::vector<double>& K, 
    std::vector<double>::iterator target_begin, std::vector<double>::iterator target_end,
    std::vector<double>::iterator source_begin, std::vector<double>::iterator source_end, 
    std::vector<double>::iterator charge_begin, 
    std::vector<double>::iterator potentials_out_begin, int dim, vector<int>& source_inds)
  {
    
    int num_targets = (target_end - target_begin) / dim;
    int num_sources = (source_end - source_begin) / dim;
    
    int block_size = 256;
    
    // Block over targets
    for (int target_ind = 0; target_ind < num_targets; target_ind+=block_size)
    {
      
      // Iterators for this block of targets
      std::vector<double>::iterator this_begin = target_begin + dim*target_ind;
      
      std::vector<double>::iterator this_end;
      int this_block_size;
      
      // how many targets in this block
      if (target_ind + block_size < num_targets)
      {
        this_end = this_begin + dim*block_size;
        this_block_size = block_size;
      }
      else 
      {
        this_end = target_end;
        this_block_size = num_targets - target_ind;
      }
      
      std::vector<double>::iterator potentials_begin = potentials_out_begin + target_ind;
     
      // Compute the kernel matrix
      kernel.Compute(this_begin, this_end, source_begin, source_end, dim, K, source_inds);
      // Apply K to the vector of charges
      double oned = 1.0;
      int one = 1;
      double zerod = 0.0;
    
      cblas_dgemv("N", &this_block_size, &num_sources, &oned, 
        K.data(), &this_block_size, &(*charge_begin), &one, &zerod, &(*potentials_begin), &one);
    
    } // loop over blocks
    
  } // ComputeKernelMatvec

  // computes K*q for a single target
  template<class TKernel>
  double ComputeKernelDotProd(TKernel& kernel, std::vector<double>& K,
    std::vector<double>& target, 
    std::vector<double>::iterator source_begin, std::vector<double>::iterator source_end,
    std::vector<double>::iterator charge_begin, int dim, vector<int>& source_inds)
  {
  
    // Compute K
    kernel.Compute(target.begin(), target.end(), source_begin, source_end, dim, K, source_inds);

    int num_sources = (source_end - source_begin) / dim;

    /*
    std::cout << "\n";
    for (int i = 0; i < num_sources; i++) {
      std::cout << K[i] << ",";
    }
    std::cout << "\n";
    */
      
    // K is 1 x k, so use dot

    //double potential = cblas_ddot(num_sources, K.data(), 1, &(*charge_begin), 1);
    int one = 1;
    double potential = cblas_ddot(&num_sources, K.data(), &one, &(*charge_begin), &one);

    

    return potential;
  
  } // ComputeKernelDotProd
  
  // Computes the minimum and maximum possible kernel result from a node
  template<class TKernel>
  std::pair<double, double> ComputeNodeBounds(TKernel& kernel, 
      std::vector<double>& query, std::vector<double>& centroid, double radius)
  {
    
    int dim = query.size();
    
    double center_dist = 0.0;
    //knn::compute_distances(centroid.data(), query.data(), 1, 1, dim, &center_dist);
    //center_dist = sqrt(center_dist);
  
    for (int i = 0; i < centroid.size(); i++)
    {
      center_dist += (centroid[i] - query[i]) * (centroid[i] - query[i]);
    }
    center_dist = sqrt(center_dist);
  
    std::vector<double> dists(2);
    double max_dist = center_dist + radius;
    dists[0] = max_dist * max_dist;
    double min_dist = std::max(0.0, center_dist - radius);
    dists[1] = min_dist * min_dist;
    
    std::vector<double> kernels;
    kernel.Compute(dists, kernels);

    //std::cout << "bounds: " << kernels[0] << ", " << kernels[1] << "\n";
    return std::pair<double, double> (kernels[0], kernels[1]);
    
  } // ComputeNodeBounds

} // namespace

#endif


