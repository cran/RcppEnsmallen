/**
 * @file eggholder_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Eggholder function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_EGGHOLDER_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_EGGHOLDER_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Eggholder function, defined by
 *
 * \f[
 * f(x) = -(x_2 + 47) * \sin(\sqrt(\left|x_2 + x_1 / 2 + 47\right|)) - x_1 *
 *    \sin(\sqrt(\left|x_1-(x_2 + 47)\right|))
 * \f]
 *
 * This should optimize to f(x) = -959.6407, at x = [512, 404.2319].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Jamil2013,
 *   title   = {A Literature Survey of Benchmark Functions For Global
 *              Optimization Problems},
 *   author  = {Momin Jamil and Xin{-}She Yang},
 *   journal = {CoRR},
 *   year    = {2013},
 *   url     = {http://arxiv.org/abs/1308.4008}
 * }
 * @endcode
 */
class EggholderFunction
{
 public:
  //! Initialize the EggholderFunction.
  EggholderFunction();

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const { return MatType("-333; -333"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("512; 404.2319"); }

  //! Get the final objective.
  double GetFinalObjective() const { return -959.6407; }

  /**
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t begin,
                                       const size_t batchSize) const;

  /**
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * Evaluate the gradient of a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   * @param batchSize Number of points to process.
   */
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const;

  /**
   * Evaluate the gradient of a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   * @param gradient The function gradient.
   */
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates, GradType& gradient);
};

} // namespace test
} // namespace ens

// Include implementation.
#include "eggholder_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_EGGHOLDER_FUNCTION_HPP
