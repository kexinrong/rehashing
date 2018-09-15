
#ifndef ASKIT_POWER_METHOD_IMPL_HPP_
#define ASKIT_POWER_METHOD_IMPL_HPP_


namespace askit {

template <class TAlg>
double PowerMethod(TAlg& alg, double acc, int max_iterations)
{

  // The algorithm has already been constructed. This calls one extra upward
  // pass, but it's simple.
  
  
  // Generate normalized vector of ones
  double scale = 1.0 / sqrt((double)alg.N);
  std::vector<double> q(alg.N, scale);
  
  double old_norm = 0.0;

  double this_norm = ComputeNorm(alg, q);
  
  int iter = 1;
  
  while ((fabs(old_norm - this_norm) / this_norm) > acc && iter <= max_iterations)
  {
    
    std::cout << "Iteration " << iter << ": Old Norm: " << old_norm << ", This Norm: " << this_norm << "\n";

    old_norm = this_norm;

    this_norm = ComputeNorm(alg, q);
    
    iter++;
  
  } // do power iterations 
  
  if (iter > max_iterations)
  {
    std::cout << "Power method did not converge!\n";
  }
  
  return this_norm;
  
} // PowerMethod


// computes the norm of Kq, also returns q as the renormalized Kq
template <class TAlg>
double ComputeNorm(TAlg& alg, std::vector<double>& q)
{
  
  // assuming q is normalized on input
  alg.UpdateCharges(q);
  
  std::vector<double> u = alg.ComputeAll();
  
  double u_norm = 0.0;
  double q_norm = 0.0;
  
  // compute the norm of the output
#pragma omp parallel for reduction(+:u_norm)
  for (int i = 0; i < u.size(); i++)
  {
    u_norm += u[i]*u[i];
  }
  // norm is squared
  u_norm = sqrt(u_norm);
  
  // now, update and rescale q
#pragma omp parallel for reduction(+:q_norm)
  for (int i = 0; i < q.size(); i++)
  {
    q_norm += q[i] * q[i];
    q[i] = u[i] / u_norm;
  }
  q_norm = sqrt(q_norm);

  std::cout << "Q norm: " << q_norm << ", U norm: " << u_norm << "\n";
  
  return u_norm;
  
}



} // namespace 


#endif
