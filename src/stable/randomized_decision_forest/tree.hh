#ifndef TREE_H
#define TREE_H

// STL includes
#include <vector>

// Local includes
#include "datasample.hh"
#include "features/tree_feature.hh"

namespace Rdfs {

class Tree
{
public:
  /**
   * @brief Default constructor
   *
   */
  Tree();

  /**
   * @brief Destructor
   *
   */
  ~Tree();

  /**
   * @brief Get the histogram of the leaf node a sample end up in.
   *        Make sure InitEvaluation was called prior to this with the correct image!
   *
   * @param x X coordinate of pixel
   * @param y Y coordinate of pixel
   *
   * @return Tree leaf node's class distribution
   */

  //const std::vector<float>& GetHistogram(unsigned int x, unsigned int y, const Utils::DataImage &data_image) const;
  const std::vector<float> GetHistogram(unsigned int x, unsigned int y) const;
  /**
   * @brief Get the number of nodes of the tree
   *
   * @return Number of nodes of the tree
   */
  unsigned int CountNodes() const;

  /**
   * @brief Get the number of leaf nodes of the tree
   *
   * @return Number of the leaf nodes of the tree
   */
  unsigned int CountLeafNodes() const;

  /**
   * @brief Get the depth of the tree
   *
   * @return Depth of the tree
   */
  unsigned int TreeDepth() const;

  /**
   * @brief Links the given left/right subtrees as children to this one. Assigns unique labels to the children nodes.
   *        Makes the current node a non-leaf node. This tree node assumes ownership of split and both children.
   *
   * @param split Split function
   * @param left Left sub-tree
   * @param right Right sub-tree
   */
  void AddChildren(TreeFeature *feature, Tree *left, Tree *right);

  /**
   * @brief Set the label of the leaf
   *
   * @param leaf_label Label to be set
   */
  void SetLabel(signed char leaf_label);

  /**
   * @brief Check if the current node is a leaf
   *
   * @return True if the current node is a leaf node, false otherwise.
   */
  bool IsLeaf() const;

  /**
   * @brief Get the left sub-tree
   *
   * @return Left sub-tree
   */
  Tree *Left();

  /**
   * @brief Get the right sub-tree
   *
   * @return Right sub-tree
   */
  Tree *Right() ;

  /**
   * @brief Get the spliting function
   *
   * @return Spliting function
   */
  const TreeFeature *Split() const;


  /**
   * @brief Update the histogram of a node.
   *
   * @param class_count Vector holding th class counts for the different labels.
   */
  void UpdateHistogram(const std::vector<int> &class_count);

  /**
   * @brief Update the histogram of a node, considering a weight which is used to combine the old histogram and the new one.
   *
   * @param class_count Vector holding th class counts for the different labels.
   * @param repass_weight Weight used to average between the old and new histograms.
   */
  void UpdateHistogram(const std::vector<int> &class_count, float repass_weight);


  /**
   * @brief Update the histogram of a node using the full training dataset.
   *
   */
  void TrainHistograms(const std::vector<Utils::DataImage> &data_images, std::vector<DataSample> &samples, float repass_weight);

  /**
   * @brief Print some tree statistics
   *
   * @return a string with some tree statistics.
   *
   */
  std::string PrintTree() const;

  int GetStatus();
  void SetStatus(int status);

  void ParseFeatures(std::vector<int> &feature_storage) const;

  void PrintLeafNodeDistributions() const;

  void PrintFeatureDistribution();

  void InitEvaluation(const Utils::DataImage& image);

  void CollectAverageLeafPatches(const std::vector<Utils::DataImage> &data_images, const std::vector<DataSample> &samples, const int radius);

  void SaveAllLeafPatch(const std::string patch_folder, const int tree_id);

  void InitAverageLeafPatches(const int radius);

  void AddImageToLeafPatch(const cv::Mat &image, const DataSample *sample);

  cv::Mat FinalizeLeafPatch();

private:
  unsigned int m_node_id;                       /// A unique node identifier within this tree
  bool m_is_leaf;                               /// If true, this node is a leaf node
  signed char m_leaf_label;                     /// If is_leaf is true, leaf_label contains the class label which had the most training instances assigned to this leaf.
  Tree *m_father;                               /// Father node or NULL if this is the root
  std::vector<float> m_histogram;               /// Class histogram
  std::vector<int> m_class_counter;             /// Class count vector, needed to compute better histograms after training.
  int m_status;                                 /// Leaf status. 0=not a leaf, 1=Too few samples, 2=No split found, 3= pure node, 4=depth based leaf.
  // If is_leaf is false, split and both left and right must be non-NULL
  // If is_leaf is true, split = left = right = NULL
  TreeFeature* m_feature; /// Spliting feature
  Tree* m_left;           /// Left sub-tree
  Tree* m_right;          /// Right sub-tree

  cv::Mat m_average_image;
  cv::Mat m_pixel_counter;
  int m_patch_counter;


  const Tree* TraverseToLeaf(unsigned int x, unsigned int y) const;

  void IncrementClassCount(unsigned int x, unsigned int y, const Utils::DataImage &data_image, unsigned int label);
  void InitHistogramTraining();
  void FinishHistogramTraining(float repass_weight);

  void PrintFeatureDistribution(std::vector<int> &feature_count);

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & m_node_id;
    ar & m_is_leaf;
    ar & m_leaf_label;
    ar & m_father;
    ar & m_histogram;
    ar & m_class_counter; // Remove if not needed at some point!
    ar & const_cast<TreeFeature * &>(m_feature);
    ar & m_left;
    ar & m_right;
  }

};
}
#endif // TREE_H
