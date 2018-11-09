/** @file kernel_independent_farfield.h
 *
 *  This file contains a header file description for the
 *  kernel-independent FMM by Lexing Ying, George Biros, and Denis
 *  Zorin.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_KERNEL_INDEPENDENT_FARFIELD_H
#define MLPACK_SERIES_EXPANSION_KERNEL_INDEPENDENT_FARFIELD_H

#include <boost/serialization/serialization.hpp>
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace mlpack {
namespace series_expansion {

/** @brief Kernel independent FMM for kernels derived from the
 *         second-order elliptic PDEs.
 */
class KernelIndependentFarField {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief The lower bound on the upward equivalent surface.
     */
    arma::vec lower_bound_upward_equivalent_;

    /** @brief The list of pseudocharges that comprise the upward
     *         equivalent density.
     */
    arma::vec pseudocharges_;

  public:

    /** @brief Serializes the far field object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & center_;
      ar & pseudocharges_;
    }

    ////////// Getters/Setters //////////

    /** @brief Gets the center of expansion.
     *
     *  @return The center of expansion for the current far-field expansion.
     */
    arma::vec &get_center();

    const arma::vec &get_center() const;

    /** @brief Gets the set of far-field coefficients.
     *
     *  @return The const reference to the vector containing the
     *          far-field coefficients.
     */
    const arma::vec& get_coeffs() const;

    /** @brief Gets the approximation order.
     *
     *  @return The integer representing the current approximation order.
     */
    short int get_order() const;

    /** @brief Gets the weight sum.
     */
    double get_weight_sum() const;

    /** @brief Sets the approximation order of the far-field expansion.
     *
     *  @param new_order The desired new order of the approximation.
     */
    void set_order(short int new_order);

    ////////// User-level Functions //////////

    /** @brief Accumulates the far field moment represented by the given
     *         reference data into the coefficients.
     */
    template<typename KernelAuxType>
    void AccumulateCoeffs(
      const KernelAuxType &kernel_aux_in,
      const arma::mat &data,
      const arma::vec &weights,
      int begin, int end, int order);

    /** @brief Refine the far field moment that has been computed before
     *         up to a new order.
     */
    template<typename KernelAuxType>
    void RefineCoeffs(
      const KernelAuxType &kernel_aux_in,
      const arma::mat &data,
      const arma::vec &weights,
      int begin, int end, int order);

    /** @brief Evaluates the far-field coefficients at the given point.
     */
    template<typename KernelAuxType>
    double EvaluateField(
      const KernelAuxType &kernel_aux_in,
      const arma::vec &point) const;

    /** @brief Initializes the current far field expansion object with
     *         the given center.
     */
    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka, const arma::vec& center);

    template<typename KernelAuxType>
    void Init(const KernelAuxType &ka);

    /** @brief Prints out the series expansion represented by this object.
     */
    template<typename KernelAuxType>
    void Print(
      const KernelAuxType &kernel_aux_in,
      const char *name = "", FILE *stream = stderr) const;

    /** @brief Translate from a far field expansion to the expansion
     *         here. The translated coefficients are added up to the
     *         ones here.
     */
    template<typename KernelAuxType>
    void TranslateFromFarField(
      const KernelAuxType &kernel_aux_in,
      const KernelIndependentFarField<ExpansionType> &se);

    /** @brief Translate to the given local expansion. The translated
     *         coefficients are added up to the passed-in local
     *         expansion coefficients.
     */
    template<typename KernelAuxType, typename KernelIndependentLocalType>
    void TranslateToLocal(
      const KernelAuxType &kernel_aux_in,
      int truncation_order, KernelIndependentLocalType *se) const;
};
}
}

#endif
