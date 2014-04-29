#ifndef TREE_FEATURE_HH
#define TREE_FEATURE_HH

// Utils includes
#include <utils/data_image.hh>

// Boost includes
#include <boost/serialization/export.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>

namespace Rdfs{

enum FeatureType {
  PIXEL_COLOR = 0,
  COLOR_ADD = 1,
  COLOR_SUB = 2,
  COLOR_SUB_ABS = 3,
  X_PIXEL_POS = 4,
  Y_PIXEL_POS = 5,
  DEPTH_FEAT = 6,
  HEIGHT = 7,
  HYBRID_SUB = 8,
  HYBRID_SUB_ABS = 9,
  HYBRID_ADD = 10,
  RELATIVE_DEPTH = 11,
  COLOR_GRADIENT = 12,
  COLOR_GRADIENT_PATCH = 13,
  GEOMETRICAL = 14,
  COLOR_GRADIENT_PATCH_COMPARISON = 15,
  COLOR_GRADIENT_PATCH_SCALED = 16,
  DEPTH_GRADIENT_PATCH = 17,
  DEPTH_GRADIENT_PATCH_SCALED = 18,
  COLOR_PATCH = 19,
  COLOR_GRADIENT_PATCH_COMPARISON_SCALED = 20,
  NORMAL_FEATURE = 21, //Always very experimental...
  ORDINAL_DEPTH = 22,
  DEPTH_NAN_CHECK = 100
};



class TreeFeature
{
public:
  /**
   * @brief Default constructor
   */
  TreeFeature(){}

  virtual ~TreeFeature(){} //Needed so that the derived class destructors are called!

  void SetThreshold(const float threshold);

  float GetThreshold() const;

  int GetFeatureType() const;

  bool Evaluate(const int x, const int y, const Utils::DataImage &data_image);

  bool Evaluate(const int x, const int y) const;

  float ExtractFeature(const int x, const int y, const Utils::DataImage &data_image);
  //Abstract from here on out.

  virtual float ExtractFeature(const int x, const int y) const = 0;

  virtual void InitEvaluation(const Utils::DataImage &data_image) = 0;

private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & m_feature_type;
    ar & m_threshold;

  }


protected:
  unsigned int m_feature_type;  /// @brief Feature type
  float m_threshold;            /// @brief Threshold used to decide the path.
  int m_image_width;            /// @brieg Width of the image which is currently evaluated.
  int m_image_height;           /// @brieg Height of the image which is currently evaluated.

};
}

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Rdfs::TreeFeature)

#endif // TREE_FEATURE_HH
