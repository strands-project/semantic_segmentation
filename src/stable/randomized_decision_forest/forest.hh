#ifndef FOREST_HH
#define FOREST_HH

// STL include
#include <vector>

// Local includes
#include "tree.hh"
#include "datasample.hh"

// Boost includes
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// Utils includes
#include <utils/configuration.hh>
#include <utils/data_image.hh>


namespace Rdfs {
class Forest
{
public:
  Forest(unsigned int num_trees, unsigned int num_samples, unsigned int sub_sample_per_tree, unsigned int num_splits, unsigned int num_thresholds, unsigned int min_sample_count, unsigned int tree_depth, unsigned int num_classes);
  Forest(const std::string& config_filename);
  ~Forest();

  void Train(const std::vector<Utils::DataImage> &data_images);

  void getAnnotation(const Utils::DataImage &data_image, cv::Mat *result_pointer, bool parallelize_internal = false) const;
  void GetUnaryPotential(const Utils::DataImage& data_image, cv::Mat* result, bool parallelize_internal) const;
  void GetClassHistogramImage(const Utils::DataImage &data_image, cv::Mat *result_pointer, bool parallelize_internal= false) const;
  void GetClassHistogramImage(const Utils::DataImage &data_image, cv::Mat * result_pointer, const cv::Mat& valid_points, bool parallelize_internal) const;

  int GetFeatureRequirements() const;

  // Serialization/Deserialization methods
  void Save(const std::string& config_filename);
  void SaveByFilename(const std::string& forest_filename);
  static Forest *Load(const std::string& config_filename);
  static Forest *LoadByFilename(const std::string& forest_filename);
  
  
  void PrintForest() const;

  void TrainInParallel(bool flag = true);


private:
  void GetSamples(std::vector<DataSample> &data_samples, const std::vector<Utils::DataImage> &data_images, unsigned int patch_radius, bool store_classweights, std::vector<double> &class_weights, bool equal_sampling = false  );
  void SubSampleTrainingdata(std::vector<DataSample> &training_samples, std::vector<DataSample *> &tree_samples);
  void SubSampleTrainingdata(std::vector<DataSample> &training_samples, std::vector<DataSample> &tree_samples);
  void ComputeNewWeights(std::vector<DataSample *> &data_samples);

  static bool SortData(const DataSample &i, const DataSample &j);

private:
  std::vector<Tree *> m_ensemble;
  unsigned int m_tree_count;
  unsigned int m_num_samples;
  unsigned int m_sub_sample_per_tree;
  unsigned int m_split_count;
  unsigned int m_thresholds_count;
  unsigned char m_min_sample_count;
  unsigned int m_tree_depth;
  unsigned int m_class_count;
  unsigned int m_patch_radius;
  unsigned int m_max_depth_check_distance;
  std::string m_config_filename;
  float m_repass_weight;
  bool m_train_trees_parallel;
  bool m_is_loaded_from_file;
  int m_leaf_patch_radius;
  std::string m_patch_image_folder;


  friend class boost::serialization::access;
  Forest();
  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar.template register_type<Tree>();
    ar & m_tree_count;
    ar & m_num_samples;
    ar & m_sub_sample_per_tree;
    ar & m_split_count;
    ar & m_thresholds_count;
    ar & m_min_sample_count;
    ar & m_tree_depth;
    ar & m_class_count;
    ar.template register_type<std::vector<Tree *> >();
    ar & m_ensemble;
    ar & m_is_loaded_from_file;
  }


};
}


#endif // FOREST_HH
