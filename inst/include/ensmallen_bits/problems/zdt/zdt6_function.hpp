/**
 * @file zdt6_function.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the sixth ZDT(Zitzler, Deb and Thiele) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ZDT_SIX_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ZDT_SIX_FUNCTION_HPP

namespace ens {
namespace test {
/**
 * The ZDT6 function, defined by:
 * \f[
 * g(x) = 1 + 9[ \sum_{i=2}^{n}(x_i^2)/9]^{0.25}
 * f_1(x) = 1 - e^{-4x_1}sin^{6}(6\pi x_1)
 * h(f1, g) = 1 - (f_1/g)^{2}
 * f_2(x) = g(x) * h(f_1, g)
 * \f]
 *
 * This is a 10-variable problem(n = 10) with a
 * non-convex optimal front. The density of the
 * solutions across optimal region is non-uniform.
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n
 *
 * This should be optimized to g(x) = 1.0, at:
 * x_1* in [0, 1] ; x_i* = 0 for i = 2,...,n
 *
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Zitzler2000,
 *   title   = {Comparison of multiobjective evolutionary algorithms:
 *              Empirical results},
 *   author  = {Zitzler, Eckart and Deb, Kalyanmoy and Thiele, Lothar},
 *   journal = {Evolutionary computation},
 *   year    = {2000},
 *   doi     = {10.1162/106365600568202}
 * }
 * @endcode
 *
 * @tparam MatType Type of matrix to optimize.
 */
  template<typename MatType = arma::mat>
  class ZDT6
  {
   private:
    size_t numParetoPoints {100};
    size_t numObjectives {2};
    size_t numVariables {10};

   public:
     //! Initialize the ZDT6
    ZDT6(size_t numParetoPoints = 100) :
        numParetoPoints(numParetoPoints),
        objectiveF1(*this),
        objectiveF2(*this)
    {/* Nothing to do here. */}

    /**
     * Evaluate the objectives with the given coordinate.
     *
     * @param coords The function coordinates.
     * @return arma::Col<typename MatType::elem_type>
     */
    arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
    {
      // Convenience typedef.
      typedef typename MatType::elem_type ElemType;

      arma::Col<ElemType> objectives(numObjectives);
      objectives(0) = 1. - std::exp(-4 * coords[0]) *
          std::pow(std::sin(6 * arma::datum::pi * coords[0]), 6);
      ElemType sum = std::pow(
          arma::accu(coords(arma::span(1, numVariables - 1), 0)) / 9, 0.25);
      ElemType g = 1. + 9. * sum;
      ElemType objectiveRatio = objectives(0) / g;
      objectives(1) = g * (1. - std::pow(objectiveRatio, 2));

      return objectives;
    }

    //! Get the starting point.
    MatType GetInitialPoint()
    {
      // Convenience typedef.
      typedef typename MatType::elem_type ElemType;

      return arma::Col<ElemType>(numVariables, 1, arma::fill::zeros);
    }

    struct ObjectiveF1
    {
      ObjectiveF1(ZDT6& zdtClass) : zdtClass(zdtClass)
      {/*Nothing to do here */}

      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        return 1. - std::exp(-4. * coords[0]) *
            std::pow(std::sin(6. * arma::datum::pi * coords[0]), 6.);
      }

      ZDT6& zdtClass;
    };

    struct ObjectiveF2
    {
      ObjectiveF2(ZDT6& zdtClass) : zdtClass(zdtClass)
      {/*Nothing to do here */}

      typename MatType::elem_type Evaluate(const MatType& coords)
      {
        typedef typename MatType::elem_type ElemType;

        size_t numVariables = zdtClass.numVariables;

        ElemType sum = std::pow(
            arma::accu(coords(arma::span(1, numVariables - 1), 0)) / 9, 0.25);
        ElemType g = 1. + 9. * sum;
        ElemType objectiveRatio = zdtClass.objectiveF1.Evaluate(coords) / g;

        return  g * (1. - std::pow(objectiveRatio, 2));
    }

      ZDT6& zdtClass;
    };

    //! Get objective functions.
    std::tuple<ObjectiveF1, ObjectiveF2> GetObjectives()
    {
      return std::make_tuple(objectiveF1, objectiveF2);
    }

    //! Get the Reference Front.
    //! Refer PR #273 Ipynb notebook to see the plot of Reference
    //! Front. The implementation has been taken from pymoo.
    arma::cube GetReferenceFront()
    {
      arma::cube front(2, 1, numParetoPoints);
      arma::vec x = arma::linspace(0.2807753191, 1, numParetoPoints);
      arma::vec y = 1 - arma::square(x);
      for (size_t idx = 0; idx < numParetoPoints; ++idx)
        front.slice(idx) = arma::vec{ x(idx), y(idx) };

      return front;
    }

    ObjectiveF1 objectiveF1;
    ObjectiveF2 objectiveF2;
  };
  } //namespace test
  } //namespace ens
#endif
