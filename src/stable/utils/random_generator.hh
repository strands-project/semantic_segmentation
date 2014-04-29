#ifndef _VO_RANDOMGENERATOR_H_
#define _VO_RANDOMGENERATOR_H_

#include <string>

#include <gsl/gsl_rng.h>

namespace Utils {
  class RandomGenerator {
  public:
    /**
     * @brief  Constructor (default)
     */
    RandomGenerator();

    /**
     * @brief  Destructor
     */
    ~RandomGenerator();

    /**
     * @brief  Equality operator for RandomGenerator objects
     * @param  other RandomGenerator object that will be assigned
     * @return New RandomGenerator object
     */
    RandomGenerator &operator=(const RandomGenerator &other);

    /**
     * @brief  Set the random seed
     * @param  seed Random seed
     */
    void Set(unsigned long int seed);

    /**
     * @brief  Sample a random integer from 0 to n-1 (if n is given), else every possible
     * @param  n Maximum integer value
     * @return Random integer
     */
    const unsigned long int Get(unsigned long int n = 0) const;

    /**
     * @brief  Get a double uniformly distributed in the range [0,1)
     * @return Double uniformly distributed in the range [0,1)
     */
    const double Uniform() const;

    /**
     * @brief Return a random variable between \f$[-1,1]\f$ with respect to an uniform distribution
     *
     *
     * @return A random variable between \f$[-1,1]\f$ with respect to an uniform distribution
     */
    const double CUniform() const;

    /**
     * @brief Sample a random value for (approximate) normal distribution with zero mean.
     *
     * @param v Variance
     *
     * @return Random sample from normal distribution with zero mean
     */
    const double SampleNormalDist(double v) const;

    /**
     * @brief Sample a random value from uniform distribution in a circle.
     *        (http://www.comnets.uni-bremen.de/itg/itgfg521/per_eval/p001.html)
     *
     * @param a Angle result \f$ [-\pi, \pi] \f$
     * @param r Radius result \f$ [-1, 1] \f$
     */
    const void SampleCircleUniformDist(double &a, double &r) const;

    /**
     * @brief  Get a double uniformly distributed in the range (0,1)
     * @return Double uniformly distributed in the range (0,1)
     */
    const double UniformPositive() const;

    /**
     * @brief  Get a random integer from 0 to n-1
     * @param  n End value
     * @return Random integer from 0 to n-1
     */
    const unsigned long int UniformInt(unsigned long int n) const;

    /**
     * @brief  Get the name of the random generator
     * @return Name of the random generator
     */
    const std::string Name() const;

    /**
     * @brief  Get the maximum value the generator can give you
     * @return Maximum value the generator can give you
     */
    const unsigned long int Max() const;

    /**
     * @brief  Get the minimum value the generator can give you
     * @return Minimum value the generator can give you
     */
    const unsigned long int Min() const;

    /**
     * @brief  Get the GSL generator
     * @return Generator
     */
    gsl_rng *GSLObj();
    const gsl_rng *GSLObj() const;

    /**
     * @brief  Random number between [0, maxEx-1] \in Z
     * @param  Maximum value
     * @return Random number
     */
    int RandTo(int max_ex);

    /**
     * @brief  Set random seed
     * @param  seed Random seed
     */
    void SetRandSeed(int seed);
  private:
    gsl_rng *m_generator; /**< GSL random number generator */
  };
}

#endif /* _VO_RANDOMGENERATOR_H_ */
