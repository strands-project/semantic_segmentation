/**
 * @file   random_source.hh
 * @author Georgios Floros <floros@vision.rwth-aachen.de>
 * @date   Thu Sep 27 11:47:53 2012
 * 
 * @brief  A class wrapping random number generation cleanly.
 *
 * @par    The code has been adapted from the tuwo library of Sebastian Nowozin
 *         (http://www.nowozin.net/sebastian/tuwo)
 */

#ifndef _UTILS_RANDOM_SOURCE_H_
#define _UTILS_RANDOM_SOURCE_H_

// C includes
#include <sys/time.h>
#include <time.h>

// Boost includes
#include <boost/random.hpp>

namespace Utils {
  class RandomSource {
  public:
    static boost::mt19937 &GlobalRandomSampler() {
      if (m_initialized == false) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	m_random_sampler.seed(tv.tv_sec + tv.tv_usec);
	m_initialized = true;
      }
      return m_random_sampler;
    }

  private:
    static boost::mt19937 m_random_sampler;
    static bool m_initialized;
  };
}

#endif /* _UTILS_RANDOM_SOURCE_H_ */
