/**
 * @file cmaes.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Definition of the Covariance Matrix Adaptation Evolution Strategy as proposed
 * by N. Hansen et al. in "Completely Derandomized Self-Adaptation in Evolution
 * Strategies".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_CMAES_HPP
#define ENSMALLEN_CMAES_CMAES_HPP

#include "full_selection.hpp"
#include "random_selection.hpp"
#include "transformation_policies/empty_transformation.hpp"
#include "transformation_policies/boundary_box_constraint.hpp"

namespace ens {

/**
 * CMA-ES - Covariance Matrix Adaptation Evolution Strategy is s a stochastic
 * search algorithm. CMA-ES is a second order approach estimating a positive
 * definite matrix within an iterative procedure using the covariance matrix.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Hansen2001
 *   author    = {Hansen, Nikolaus and Ostermeier, Andreas},
 *   title     = {Completely Derandomized Self-Adaptation in Evolution
 *                Strategies},
 *   journal   = {Evol. Comput.},
 *   volume    = {9},
 *   number    = {2},
 *   year      = {2001},
 *   pages     = {159--195},
 *   publisher = {MIT Press},
 * }
 * @endcode
 *
 * CMA-ES can optimize separable functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam SelectionPolicy The selection strategy used for the evaluation step.
 * @tparam TransformationPolicy The transformation strategy used to 
 *       map decision variables to the desired domain during fitness evaluation
 *       and termination. Use EmptyTransformation if the domain isn't bounded.
 */
template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>>
class CMAES
{
 public:
  /**
   * Construct the CMA-ES optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param lambda The population size (0 use the default size).
   * @param transformationPolicy Instantiated transformation policy used to 
   *     map the coordinates to the desired domain.
   * @param batchSize Batch size to use for the objective calculation.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param selectionPolicy Instantiated selection policy used to calculate the
   *     objective.
   * @param stepSize Starting sigma/step size (will be modified).
   */
  CMAES(const size_t lambda = 0,
        const TransformationPolicyType& 
              transformationPolicy = TransformationPolicyType(),
        const size_t batchSize = 32,
        const size_t maxIterations = 1000,
        const double tolerance = 1e-5,
        const SelectionPolicyType& selectionPolicy = SelectionPolicyType(),
        double stepSize = 0);

  /**
   * Construct the CMA-ES optimizer with the given function and parameters 
   * (including lower and upper bounds). The defaults here are not necessarily 
   * good for the given problem, so it is suggested that the values used be 
   * tailored to the task at hand.  The maximum number of iterations refers to 
   * the maximum number of points that are processed (i.e., one iteration 
   * equals one point; one iteration does not equal one pass over the dataset).
   *
   * @param lambda The population size(0 use the default size).
   * @param lowerBound Lower bound of decision variables.
   * @param upperBound Upper bound of decision variables.
   * @param batchSize Batch size to use for the objective calculation.
   * @param maxIterations Maximum number of iterations allowed(0 means no
      limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param selectionPolicy Instantiated selection policy used to calculate the
   * objective.
   * @param stepSize Starting sigma/step size (will be modified).
   */
  CMAES(const size_t lambda = 0,
        const double lowerBound = -10,
        const double upperBound = 10,
        const size_t batchSize = 32,
        const size_t maxIterations = 1000,
        const double tolerance = 1e-5,
        const SelectionPolicyType& selectionPolicy = SelectionPolicyType(),
        double stepSize = 0);

  /**
   * Optimize the given function using CMA-ES. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam SeparableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename SeparableFunctionType,
      typename MatType,
      typename... CallbackTypes>
      typename MatType::elem_type Optimize(SeparableFunctionType& function,
          MatType& iterate,
          CallbackTypes&&... callbacks);

  //! Get the population size.
  size_t PopulationSize() const { return lambda; }
  //! Modify the population size.
  size_t& PopulationSize() { return lambda; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get the selection policy.
  const SelectionPolicyType& SelectionPolicy() const { return selectionPolicy; }
  //! Modify the selection policy.
  SelectionPolicyType& SelectionPolicy() { return selectionPolicy; }

  //! Get the transformation policy.
  const TransformationPolicyType& TransformationPolicy() const
  { return transformationPolicy; }
  //! Modify the transformation policy.
  TransformationPolicyType& TransformationPolicy() 
  { return transformationPolicy; }

  //! Get the step size.
  double StepSize() const
  { return stepSize; }
  //! Modify the step size.
  double& StepSize()
  { return stepSize; }

  //! Get the total number of function evaluations.
  size_t FunctionEvaluations() const  { return functionEvaluations; }

 private:
  //! Population size.
  size_t lambda;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The selection policy used to calculate the objective.
  SelectionPolicyType selectionPolicy;

  //! The transformationPolicy used to map coordinates to the suitable domain
  //! while evaluating fitness. This mapping is also done after optimization 
  //! has completed.
  TransformationPolicyType transformationPolicy;

  //! The step size.
  double stepSize;

  //! Counter for the number of function evaluations.
  size_t functionEvaluations = 0;
};

/**
 * Convenient typedef for CMAES approximation.
 */
template<typename TransformationPolicyType = EmptyTransformation<>,
         typename SelectionPolicyType = RandomSelection>
using ApproxCMAES = CMAES<SelectionPolicyType, TransformationPolicyType>;

} // namespace ens

// Include implementation.
#include "cmaes_impl.hpp"

#endif
