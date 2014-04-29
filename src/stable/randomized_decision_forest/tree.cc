//Local includes
#include "tree.hh"

// STL includes
#include <sstream>

// Boost includes
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/progress.hpp>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace Rdfs;
Tree::Tree(): m_node_id(0), m_is_leaf(true), m_leaf_label(0), m_father(NULL), m_feature(NULL), m_left(NULL), m_right(NULL) {
}

Tree::~Tree() {
  if (m_is_leaf) {
    return;
  }

  delete(m_feature);
  delete(m_left);
  delete(m_right);
}


const std::vector<float> Tree::GetHistogram(unsigned int x, unsigned int y) const{
  return (TraverseToLeaf(x, y)->m_histogram);
}

unsigned int Tree::CountNodes() const {
  return (m_is_leaf ? 1 : (1 + m_left->CountNodes() + m_right->CountNodes()));
}

unsigned int Tree::CountLeafNodes() const{
  return (m_is_leaf ? 1 : (m_left->CountLeafNodes() + m_right->CountLeafNodes()));
}


unsigned int Tree::TreeDepth() const{
  return (m_is_leaf ? 1 : (1 + std::max(m_left->TreeDepth(), m_right->TreeDepth())));
}

void Tree::AddChildren(TreeFeature *feature, Tree *left, Tree *right) {
  assert(m_is_leaf == true);
  assert(left->m_node_id == 0);
  assert(right->m_node_id == 0);

  // Find root and obtain some unique id's, continuously
  Tree *root = this;
  while (root->m_father != NULL) {
    root = root->m_father;
  }
  unsigned int node_count = root->CountNodes();

  left->m_node_id = node_count;
  right->m_node_id = node_count + 1;

  // Link
  this->m_left = left;
  this->m_right = right;
  left->m_father = this;
  right->m_father = this;
  this->m_feature = feature;

  // This node becomes interior to the tree
  m_is_leaf = false;
}

void Tree::SetLabel(signed char leaf_label) {
  this->m_leaf_label = leaf_label;
}


bool Tree::IsLeaf() const{
  return (m_is_leaf);
}

Tree *Tree::Left(){
  assert(m_is_leaf == false);
  return (m_left);
}

Tree *Tree::Right(){
  assert(m_is_leaf == false);
  return (m_right);
}


const TreeFeature *Tree::Split() const {
  return m_feature;
}

void Tree::UpdateHistogram(const std::vector<int> &class_count, float repass_weight) {
  int total_samples = 0;
  for(unsigned int i=0; i< class_count.size(); ++i){
    total_samples += class_count[i];
  }
  std::vector<double> new_hist(class_count.size(), 0);

  //The unlucky case that no data ended up in this node
  if(total_samples==0){
    for(unsigned int i=0; i< class_count.size(); ++i){
      new_hist[i] = 1.0f/class_count.size();
    }
  }else{
    for(unsigned int i=0; i< class_count.size(); ++i){
      new_hist[i] = static_cast<double>(class_count[i]) / static_cast<double>(total_samples);
    }
  }
  float old_weight = 1.0f - repass_weight;
  for(unsigned int i=0; i< class_count.size(); ++i){
    m_histogram[i] = m_histogram[i]*old_weight + new_hist[i]*repass_weight;
  }

}

void Tree::UpdateHistogram(const std::vector<int> &class_count) {
  int total_samples = 0;
  for(unsigned int i=0; i< class_count.size(); ++i){
    total_samples += class_count[i];
  }
  m_histogram.clear();
  m_histogram.resize(class_count.size());
  //The unlucky case that  no data ended up in this node
  if(total_samples==0){
    for(unsigned int i=0; i< class_count.size(); ++i){
      m_histogram[i] = 1.0f/class_count.size();
    }
  }else{
    for (unsigned int i=0; i< class_count.size(); ++i){
      m_histogram[i] = static_cast<double>(class_count[i]) / static_cast<double>(total_samples);
    }
  }
}

void Tree::TrainHistograms(const std::vector<Utils::DataImage> &data_images,  std::vector<DataSample> &samples, float repass_weight){
  //For each training sample, traverse the tree
  //and update the according class bin in the class counter vector.
  const int size = samples.size();
  //Resize all the vectors to fit the labels.
  InitHistogramTraining();
  std::cout << "(train) Passing all training points through the tree..." << std::endl;
  //Pass the data through the tree.

  boost::progress_display progress(size);
  for(int i=0; i < size; ++i){
    const DataSample sample = samples[i];
    IncrementClassCount(sample.x_pos,sample.y_pos, data_images[sample.image_id], static_cast<unsigned int>(sample.center_label));
    ++progress;
  }
  //Normalize
  FinishHistogramTraining(repass_weight);

}


std::string Tree::PrintTree() const{
  std::stringstream description;
  description << "Tree with " << CountNodes() << " nodes, " << CountLeafNodes()<<" leaf nodes and depth: " << TreeDepth() << std::endl;
  return description.str();
}

int Tree::GetStatus(){
  return m_status;
}

void Tree::SetStatus(int status){
  m_status = status;
}

void Tree::ParseFeatures(std::vector<int> &feature_storage) const{
  if(!m_is_leaf){
    feature_storage.push_back(m_feature->GetFeatureType());
    if(m_left != NULL){
      m_left->ParseFeatures(feature_storage);
    }
    if(m_right != NULL){
      m_right->ParseFeatures(feature_storage);
    }
  }
}

void Tree::PrintLeafNodeDistributions() const{
  if(m_is_leaf){
    std::cout << "ID: " << m_node_id << " |  ";
    for( std::vector<float>::const_iterator i = m_histogram.begin(); i != m_histogram.end(); ++i) std::cout << *i << ' ';
    std::cout <<std::endl;
  }else{
    if(m_left != NULL){
      m_left->PrintLeafNodeDistributions();
    }
    if(m_right != NULL){
      m_right->PrintLeafNodeDistributions();
    }
  }
}


void Tree::PrintFeatureDistribution(){
  std::vector<int> feature_count(200,0);
  feature_count[m_feature->GetFeatureType()] = feature_count[m_feature->GetFeatureType()] +1;

  if(m_left != NULL){
    m_left->PrintFeatureDistribution(feature_count);
  }
  if(m_right != NULL){
    m_right->PrintFeatureDistribution(feature_count);
  }
  int total = 0;
  for(unsigned int i=0; i < feature_count.size(); ++i){
    total += feature_count[i];
  }
  std::cout << "In total there are " << total << " splits." << std::endl;

  for(unsigned int i=0; i < feature_count.size(); ++i){
    if(feature_count[i]!=0){
      std::cout << "Feature " << ((i==100) ? -1 : i) << ": " << static_cast<float>(feature_count[i]*100)/total << "%" << std::endl;
    }
  }
}

void Tree::InitEvaluation(const Utils::DataImage &image){
  if(!m_is_leaf){
    m_feature->InitEvaluation(image);
    assert(m_left != NULL); //As this is not a leaf, both asserts should always pass without problems.
    assert(m_right != NULL);
    m_left->InitEvaluation(image);
    m_right->InitEvaluation(image);
  }
}

void Tree::CollectAverageLeafPatches(const std::vector<Utils::DataImage> &data_images, const std::vector<DataSample> &samples, const int radius){
  //Initialize the images in the leafes.
  InitAverageLeafPatches(radius);

  //Start filling them, this assumes the list of datasamples is sorted, based on the images they are comming from.
  int image = -1;
  // Take each sample.
  for(std::vector<DataSample>::const_iterator it = samples.begin() ; it != samples.end(); ++it){
    if(image != it->image_id){
      //We need to load the next data image.
      image = it->image_id;
      InitEvaluation(data_images[image]);
    }
    AddImageToLeafPatch(data_images[image].GetRGBImage(), &(*it));
  }
}

void Tree::SaveAllLeafPatch(const std::string patch_folder, const int tree_id){
  // Go over all leaf nodes and save it do a specific folder, with treeId_leafId.png
  if(m_is_leaf){
    cv::Mat patch = FinalizeLeafPatch();

    //Save the image
    std::stringstream filename;
    filename << patch_folder << "/" << tree_id << "_" << m_node_id << ".png";
    std::cout << filename.str() << "  "<< m_patch_counter << std::endl;
    cv::imwrite(filename.str(), patch);
  }else{
    //Traverse to leafs.
    m_left->SaveAllLeafPatch(patch_folder, tree_id);
    m_right->SaveAllLeafPatch(patch_folder, tree_id);
  }
}

void Tree::InitAverageLeafPatches(const int radius){
if (m_is_leaf) {
  m_average_image = cv::Mat(radius*2+1,radius*2+1,CV_32FC3, cv::Scalar(0,0,0));
  m_pixel_counter = cv::Mat(radius*2+1,radius*2+1,CV_32SC1, cv::Scalar(0,0,0));
  m_patch_counter=0;
}else{
  m_left->InitAverageLeafPatches(radius);
  m_right->InitAverageLeafPatches(radius);
}

}

void Tree::AddImageToLeafPatch(const cv::Mat &image, const DataSample* sample){
  if(m_is_leaf){
    m_patch_counter++;
    int x_local = 0;
    int y_local = 0;
    int radius = (m_pixel_counter.cols-1)/2;
    for(int y = sample->y_pos - radius; y <= sample->y_pos + radius; y++){
      for(int x = sample->x_pos - radius; x <= sample->x_pos + radius; x++){
        if(x >= 0 && x < image.cols && y >= 0 && y < image.rows){
          x_local = x - sample->x_pos + radius;
          y_local = y - sample->y_pos + radius;
          m_pixel_counter.at<int>(y_local, x_local)++;
          m_average_image.at<cv::Vec3f >(y_local, x_local)[0] += image.at<cv::Vec3b>(y,x)[0];
          m_average_image.at<cv::Vec3f>(y_local, x_local)[1] += image.at<cv::Vec3b>(y,x)[1];
          m_average_image.at<cv::Vec3f>(y_local, x_local)[2] += image.at<cv::Vec3b>(y,x)[2];
        }
      }
    }
  }else{
    m_feature->Evaluate(sample->x_pos,sample->y_pos)  ? m_left->AddImageToLeafPatch(image, sample) : m_right->AddImageToLeafPatch(image, sample);
  }
}

cv::Mat Tree::FinalizeLeafPatch(){
  cv::Mat patch(m_average_image.cols, m_average_image.rows, CV_8UC3);
  for(int y = 0; y < patch.rows; y++){
    for(int x = 0; x < patch.cols; x++){
      patch.at<cv::Vec3b>(y,x)[0] = m_average_image.at<cv::Vec3f>(y,x)[0]/m_pixel_counter.at<int>(y,x);
      patch.at<cv::Vec3b>(y,x)[1] = m_average_image.at<cv::Vec3f>(y,x)[1]/m_pixel_counter.at<int>(y,x);
      patch.at<cv::Vec3b>(y,x)[2] = m_average_image.at<cv::Vec3f>(y,x)[2]/m_pixel_counter.at<int>(y,x);
    }
  }
  return patch;
}

void Tree::PrintFeatureDistribution(std::vector<int> &feature_count){
  if(!m_is_leaf){
    feature_count[m_feature->GetFeatureType()] = feature_count[m_feature->GetFeatureType()] +1;

    if(m_left != NULL){
      m_left->PrintFeatureDistribution(feature_count);
    }
    if(m_right != NULL){
      m_right->PrintFeatureDistribution(feature_count);
    }
  }
}

const Tree* Tree::TraverseToLeaf(unsigned int x, unsigned int y) const{
  if (m_is_leaf) {
    return (this);
  }
  return (m_feature->Evaluate(x,y)  ? m_left->TraverseToLeaf(x,y) : m_right->TraverseToLeaf(x,y));
}

void Tree::IncrementClassCount(unsigned int x, unsigned int y, const Utils::DataImage &data_image, unsigned int label){
  if (m_is_leaf) {
    m_class_counter[label]++;
    return;
  }
  return (m_feature->Evaluate(x,y, data_image)  ? m_left->IncrementClassCount(x,y,data_image, label) : m_right->IncrementClassCount(x,y, data_image, label));


}

void Tree::InitHistogramTraining(){
  if(m_is_leaf){
    m_class_counter.resize(m_histogram.size(), 0);
  }else{
    m_left->InitHistogramTraining();
    m_right->InitHistogramTraining();
  }
}

void Tree::FinishHistogramTraining(float repass_weight){
  if(m_is_leaf){
    UpdateHistogram(m_class_counter, repass_weight);
  }else{
    m_left->FinishHistogramTraining(repass_weight);
    m_right->FinishHistogramTraining(repass_weight);
  }
}
