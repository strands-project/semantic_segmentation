//Local include
#include "tree_train.hh"

// Boost includes
#include <boost/serialization/vector.hpp>

// STL includes
#include <iostream>
#include <algorithm>

// Utils includes
#include <utils/random_source.hh>


using namespace Rdfs;

TreeTrain::TreeTrain(const std::string& config_filename){
  Utils::Configuration configuration(config_filename);
  m_config_filename = config_filename;
  m_split_count = configuration.read<unsigned int>("num_splits");
  m_thresholds_count = configuration.read<unsigned int>("num_thresholds");
  m_min_sample_count = configuration.read<unsigned int>("min_sample_count");
  m_max_tree_depth = configuration.read<unsigned int>("tree_depth");
  m_class_count = configuration.read<unsigned int>("num_classes");
  m_patch_radius = configuration.read<unsigned int>("patch_radius");
  m_max_depth_check_distance = configuration.read<unsigned int>("max_depth_check_distance");
  m_use_depth_no_depth = configuration.read<bool>("forest_use_raw");
  m_gradient_parameters.resize(4,0);
  m_gradient_parameters[0] =  configuration.read<unsigned int>("gradient_angle_bins");
  m_gradient_parameters[1] =  configuration.read<unsigned int>("gradient_patch_min");
  m_gradient_parameters[2] =  configuration.read<unsigned int>("gradient_patch_max");
  m_gradient_parameters[3] =  configuration.read<unsigned int>("gradient_patch_dist");
  m_console_tree = new Utils::Console_Tree();

}


TreeTrain::~TreeTrain(){
    delete m_console_tree;
}


Tree *TreeTrain::TrainTree(const std::vector<Utils::DataImage> &data_images, std::vector<DataSample> &samples,
                                           std::vector<DataSample *> &tree_samples){

  //Load the features we will use.
  Utils::Configuration configuration(m_config_filename);
  if(m_use_depth_no_depth){
    //We want to load two different pools. One for the nodes with depth and one for those without depth.
    unsigned int feature_count_depth = configuration.read<unsigned int>("forest_feature_count");
    unsigned int feature_count_no_depth = configuration.read<unsigned int>("forest_no_depth_count");
    std::vector<int> feature_pool_depth = Utils::Configuration::ParseIntVectorFromString(configuration.read<std::string>("forest_feature_pool"));
    std::vector<int> feature_pool_no_depth = Utils::Configuration::ParseIntVectorFromString(configuration.read<std::string>("forest_no_depth_pool"));

    while(m_features_depth.size() < feature_count_depth && feature_pool_depth.size() != 0){
      boost::uniform_int<unsigned int> uniform_dist(0, feature_pool_depth.size()-1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_ft(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);
      int index = rgen_ft();
      m_features_depth.push_back(feature_pool_depth.at(index));
      feature_pool_depth.erase(feature_pool_depth.begin() + index);
    }

    while(m_features_no_depth.size() < feature_count_no_depth && feature_pool_no_depth.size() != 0){
      boost::uniform_int<unsigned int> uniform_dist(0, feature_pool_no_depth.size()-1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_ft(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);
      int index = rgen_ft();
      m_features_no_depth.push_back(feature_pool_no_depth.at(index));
      feature_pool_no_depth.erase(feature_pool_no_depth.begin() + index);
    }

    std::cout << "features used to train this tree's depth side: ";
    for( std::vector<int>::const_iterator i = m_features_depth.begin(); i != m_features_depth.end(); ++i)
      std::cout << *i << ' ';
    std::cout << std::endl;
    std::cout << "features used to train this tree's side without depth: ";
    for( std::vector<int>::const_iterator i = m_features_no_depth.begin(); i != m_features_no_depth.end(); ++i)
      std::cout << *i << ' ';
    std::cout << std::endl;

  }else{
    //We only load one pool for nodes with depth.
    unsigned int feature_count_depth = configuration.read<unsigned int>("forest_feature_count");
    std::vector<int> feature_pool_depth = Utils::Configuration::ParseIntVectorFromString(configuration.read<std::string>("forest_feature_pool"));
    std::vector<int> feature_weights_depth = Utils::Configuration::ParseIntVectorFromString(configuration.read<std::string>("forest_feature_weights"));
    int pool_size = feature_pool_depth.size();
    for(int f=0; f< pool_size; ++f){
      for(int w=1; w < feature_weights_depth[f]; ++w){
        feature_pool_depth.push_back(feature_pool_depth[f]);
      }
    }
    while(m_features_depth.size() < feature_count_depth && feature_pool_depth.size() != 0){
      boost::uniform_int<unsigned int> uniform_dist(0, feature_pool_depth.size()-1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_ft(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);
      int index = rgen_ft();
      m_features_depth.push_back(feature_pool_depth.at(index));
      feature_pool_depth.erase(feature_pool_depth.begin() + index);
    }

    std::cout << "features used to train this tree: ";
    for( std::vector<int>::const_iterator i = m_features_depth.begin(); i != m_features_depth.end(); ++i)
      std::cout << *i << ' ';
    std::cout << std::endl;

  }

  Tree * result = Train(data_images, tree_samples);
  result->PrintFeatureDistribution();

  return result;
}


Tree *TreeTrain::Train(const std::vector<Utils::DataImage> &data_images, std::vector<DataSample *> &samples, Tree *tree_base, int depth_left, bool has_depth){
  // Create new tree, if necessary
  Tree *current_root = tree_base;
  bool is_root = false;
  if (current_root == NULL) {
    current_root = new Tree();
    m_split_generator = SplitFunctionGenerator<Utils::DataImage>(&data_images,  m_split_count, m_thresholds_count, m_class_count);
    depth_left = m_max_tree_depth;
    m_console_tree->Init(m_max_tree_depth);
    m_console_tree->PrintRoot();
    is_root = true;
  }

  // Estimate maximum label at this node
  std::vector<int> class_count(m_class_count);
  for (std::vector<DataSample *>::iterator itr= samples.begin(); itr != samples.end(); ++itr) {
    //class_count[(*itr)->label] += (*itr)->histogram_weight;
    class_count[(*itr)->center_label] += 1;
  }
  signed char max_class_id = 0;
  signed char min_class_id = 0;
  unsigned int max_class_count = 0;
  unsigned int min_class_count = std::numeric_limits<unsigned char>::infinity();
  for (int i=0; i < m_class_count; ++i){
    const unsigned int current_class_count = class_count[i];
    if(current_class_count > max_class_count){
      max_class_id = i;
      max_class_count = current_class_count;
    }
    if(current_class_count !=0 && current_class_count < min_class_count){
      min_class_id = i;
      min_class_count = current_class_count;
    }
  }
  current_root->SetLabel(max_class_id);

  // If the node is pure: leaf node
  if( min_class_id == max_class_id){
    current_root->UpdateHistogram(class_count);
    current_root->SetStatus(3);
    return (current_root);
  }

  // Number of samples below threshold: leaf node
  if (samples.size() <= m_min_sample_count) {
    current_root->UpdateHistogram(class_count);
    current_root->SetStatus(1);
    return (current_root);
  }

  // Depth exhausted: leaf node
  depth_left -= 1;
  if (depth_left == 0) {
    current_root->UpdateHistogram(class_count);
    current_root->SetStatus(4);
    return (current_root);
  }

  //Checking is done, try to find a new split function.
  if(m_use_depth_no_depth && is_root){
    //Make a deterministic choice here.
    std::vector<DataSample *> left_samples;
    std::vector<DataSample *> right_samples;

    //Create a new feature
    TreeFeature * feature= new DepthCheckFeature();

    //Create the split
    feature->SetThreshold(0);

    //Sort out left and right samples.
    for(int i=0; i < samples.size(); ++i){
      DataSample* current = samples[i];
      if(feature->Evaluate(current->x_pos, current->y_pos, data_images[current->image_id])){
        left_samples.push_back(current); //depth
      }else{
        right_samples.push_back(current); //no depth
      }
    }

    if(right_samples.size() ==0 || left_samples.size() == 0) {
      m_console_tree->Update(100);
      current_root->UpdateHistogram(class_count);
      current_root->SetStatus(2);
      return (current_root);
    }
    m_console_tree->Update(100, -1);

    Tree *left = new Tree();
    Tree *right = new Tree();
    current_root->AddChildren(feature, left, right);

    m_console_tree->PrintLeftChild(depth_left);
    Train(data_images,left_samples, left, depth_left, true);
    m_console_tree->PrintRightChild(depth_left, left->GetStatus());
    Train(data_images, right_samples, right, depth_left, false);
    m_console_tree->PopBack(depth_left, right->GetStatus());

  }else{
    // Sample features and find the best feature and threshold.
    std::vector<DataSample *> left_samples;
    std::vector<DataSample *> right_samples;
    TreeFeature* split = m_split_generator.GetSplitFunction(samples, left_samples, right_samples, m_patch_radius, m_max_depth_check_distance, m_gradient_parameters, m_console_tree, has_depth ? m_features_depth : m_features_no_depth, m_max_tree_depth - depth_left);
    if(split == NULL) {
      m_console_tree->Update(100);
      current_root->UpdateHistogram(class_count);
      current_root->SetStatus(2);
      return (current_root);
    }
    m_console_tree->Update(100, split->GetFeatureType());
    // We found a good enough split, grow the tree.  The current_root
    // assumes ownership of split, left and right.
    Tree *left = new Tree();
    Tree *right = new Tree();
    current_root->AddChildren(split, left, right);

    m_console_tree->PrintLeftChild(depth_left);
    Train(data_images,left_samples, left, depth_left, has_depth);
    m_console_tree->PrintRightChild(depth_left, left->GetStatus());
    Train(data_images, right_samples, right, depth_left, has_depth);
    m_console_tree->PopBack(depth_left, right->GetStatus());

  }
  return(current_root);
}
