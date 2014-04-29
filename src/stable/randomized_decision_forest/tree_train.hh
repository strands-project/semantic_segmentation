#ifndef TREE_TRAIN_HH
#define TREE_TRAIN_HH

// Local includes
#include "tree.hh"
#include "split_function_generator.hh"

// Utils includes
#include <utils/configuration.hh>
#include <utils/console_tree.hh>
#include <utils/data_image.hh>

// STL includes
#include <vector>


namespace Rdfs{
/**
 * @class TreeTrain
 * @brief A class encapsulating the training of a randomized decision tree.
 */
class TreeTrain
{
public:
  TreeTrain(const std::string& config_filename);

  ~TreeTrain();
  /**
   * @brief Train a randomized decision tree
   *
   * @param image_pointers Pointers to the RGB images in LAB color space
   * @param depth_image_pointers a vector of points to the depth images
   * @param samples a vector of chosen samples for the whole forest.
   * @param tree_samples  a vector of poiters to some of the sampels used for training this tree.
   * @param train_trees_parallel boolean which specifies if the tree is being trained in parallel to other trees.
   *
   * @return Trained randomized decision tree.
   */
  Tree *TrainTree(const std::vector<Utils::DataImage> &data_images, std::vector<DataSample> &samples, std::vector<DataSample *> &tree_samples);

private:

  Tree *Train(const std::vector<Utils::DataImage> &data_images, std::vector<DataSample *> &samples, Tree *tree_base = NULL, int depth_left =-1, bool has_depth = true);

  unsigned int            m_sample_count;        /// @brief Number of samples used for training.
  unsigned int            m_split_count;         /// @brief Number of splits
  unsigned int            m_thresholds_count;    /// @brief Number of thresholds per split
  unsigned int            m_min_sample_count;    /// @brief Minimum number of samples
  unsigned int            m_max_tree_depth;      /// @brief Maximum tree depth
  unsigned char           m_class_count;         /// @brief Number of labels
  unsigned int            m_patch_radius;         /// @brief Patch radius used for training.
  Utils::Console_Tree*    m_console_tree;        /// @brief For visualizing the trees
  SplitFunctionGenerator<Utils::DataImage>  m_split_generator;      /// @brief This object is used to generate new splits.
  unsigned int            m_max_depth_check_distance;  /// @brief Used for depth features.
  std::vector<unsigned int>   m_gradient_parameters;             /// @brief Holds 4 gradient parameters: 1. gradient channels 2. min gradient patch size 3. max gradient patch size 4. max gradient patch distance.
  std::vector<int>        m_features_depth;            /// @brief features used to train this trees depth side.
  std::vector<int>        m_features_no_depth;         /// @brief features used to train this trees side without depth.
  std::string             m_config_filename;
  bool                    m_use_depth_no_depth;
};
}
#endif // TREE_TRAIN_HH
