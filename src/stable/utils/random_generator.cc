#include "random_generator.hh"

// C includes
#include <math.h>

using namespace Utils;

RandomGenerator::RandomGenerator() {
  m_generator = NULL;
  gsl_rng_env_setup();
  m_generator = gsl_rng_alloc(gsl_rng_default);
}

RandomGenerator::~RandomGenerator() {
  gsl_rng_free(m_generator);
}

RandomGenerator &RandomGenerator::operator=(const RandomGenerator &other) {
  if (m_generator) {
    gsl_rng_free(m_generator);
  }
  m_generator = gsl_rng_clone(other.m_generator);
  return *this;
}

void RandomGenerator::Set(unsigned long int seed) {
  gsl_rng_set(m_generator, seed);
}

const unsigned long int RandomGenerator::Get(unsigned long int n) const {
  if (n) {
    return gsl_rng_uniform_int(m_generator, n);
  } else {
    return gsl_rng_get(m_generator);
  }
}

const double RandomGenerator::Uniform() const {
  return gsl_rng_uniform(m_generator);
}

const double RandomGenerator::CUniform() const {
  return 1.0 - (2.0 * Uniform());
}

const double RandomGenerator::SampleNormalDist(double v) const {
  double sum = 0;
  for (int i = 0; i < 12; i++) {
    sum += (CUniform() * v);
  }
  return sum / 2.0;
}

const void RandomGenerator::SampleCircleUniformDist(double &a, double &r) const {
  a = CUniform() * M_PI;
  r = sqrt(Uniform());
}

const double RandomGenerator::UniformPositive() const {
  return gsl_rng_uniform_pos(m_generator);
}

const unsigned long int RandomGenerator::UniformInt(unsigned long int n) const {
  return gsl_rng_uniform_int(m_generator, n);
}

const std::string RandomGenerator::Name() const {
  return gsl_rng_name(m_generator);
}

const unsigned long int RandomGenerator::Max() const {
  return gsl_rng_max(m_generator);
}

const unsigned long int RandomGenerator::Min() const {
  return gsl_rng_min(m_generator);
}

gsl_rng *RandomGenerator::GSLObj() {
  return m_generator;
}

const gsl_rng *RandomGenerator::GSLObj() const {
  return m_generator;
}

int RandomGenerator::RandTo(int max_ex) {
  return Get(max_ex);
}

void RandomGenerator::SetRandSeed(int seed) {
  Set(seed);
}
