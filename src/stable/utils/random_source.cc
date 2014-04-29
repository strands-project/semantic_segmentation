/**
 * @file   random_source.cc
 * @author Georgios Floros <floros@vision.rwth-aachen.de>
 * @date   Thu Sep 27 11:52:12 2012
 * 
 * @brief  A class wrapping random number generation cleanly. (Implementation)
 *
 * @par    The code has been adapted from the tuwo library of Sebastian Nowozin
 *         (http://www.nowozin.net/sebastian/tuwo)
 */

#include "random_source.hh"

namespace Utils {
  // Global static variable
  bool RandomSource::m_initialized = false;
  boost::mt19937 RandomSource::m_random_sampler;
}
