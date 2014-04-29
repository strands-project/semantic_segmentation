// Local includes
#include "forest.hh"

// STL include
#include <vector>
#include <sstream>
#include <algorithm>
#include <fstream>

// Local includes
#include "tree_train.hh"

// Utils includes
#include <utils/random_source.hh>
#include <utils/dataset.hh>

// OpenCV includes
#include <opencv2/core/core.hpp>

using namespace Rdfs;

Forest::Forest(unsigned int num_trees, unsigned int num_samples, unsigned int sub_sample_per_tree, unsigned int num_splits,
               unsigned int num_thresholds, unsigned int min_sample_count, unsigned int tree_depth, unsigned int num_classes):
  m_tree_count(num_trees), m_num_samples(num_samples), m_sub_sample_per_tree(sub_sample_per_tree), m_split_count(num_splits),
  m_thresholds_count(num_thresholds), m_min_sample_count(min_sample_count), m_tree_depth(tree_depth), m_class_count(num_classes) {
  m_ensemble.resize(m_tree_count);
}

Forest::Forest(const std::string& config_filename){
  Utils::Configuration configuration(config_filename);
  m_config_filename = config_filename;
  m_tree_count = configuration.read<unsigned int>("num_trees");
  m_num_samples = configuration.read<unsigned int>("num_samples");
  m_sub_sample_per_tree = configuration.read<unsigned int>("sub_sample_per_tree");
  m_split_count = configuration.read<unsigned int>("num_splits");
  m_thresholds_count = configuration.read<unsigned int>("num_thresholds");
  m_min_sample_count = configuration.read<unsigned int>("min_sample_count");
  m_tree_depth = configuration.read<unsigned int>("tree_depth");
  m_class_count = configuration.read<unsigned int>("num_classes");
  m_patch_radius = configuration.read<unsigned int>("patch_radius");
  m_max_depth_check_distance = configuration.read<unsigned int>("max_depth_check_distance");
  m_repass_weight = configuration.read<float>("repass_weight");
  m_leaf_patch_radius = configuration.read<int>("leaf_patch_radius", 10);
  m_patch_image_folder = configuration.read<std::string>("leaf_patch_folder");
  m_ensemble.resize(m_tree_count);
  m_train_trees_parallel = true;
  m_is_loaded_from_file = false;
}

Forest::Forest(): m_tree_count(0), m_split_count(0), m_thresholds_count(0),
  m_min_sample_count(0), m_tree_depth(0), m_class_count(0) {
}

Forest::~Forest(){
//  std::cout << "Deleting the forest"<<std::endl;
  for(std::vector<Tree *>::const_iterator itr = m_ensemble.begin(); itr != m_ensemble.end(); ++itr ){
    delete(*itr);
  }
}

void Forest::Train(const std::vector<Utils::DataImage> &data_images){
  std::cout << "Learning image tree ensemble (T=" << m_tree_count << ")..." << std::endl;
  //Subsample the data randomly.
  std::vector<DataSample> samples;
  std::vector<double> class_weights;
  GetSamples(samples, data_images, m_patch_radius, true, class_weights, false);

  //Setup and train all the trees

#pragma omp parallel for if(m_train_trees_parallel)
  for (int t = 0; t < static_cast<int>(m_tree_count); ++t) {

    //Setup the training
    TreeTrain tree_train = TreeTrain(m_config_filename);

    //For each tree, subsample the training set randomly based on sub_sample_per_tree.
    if(m_train_trees_parallel){
      std::vector<DataSample > tree_samples_storage;
      SubSampleTrainingdata(samples, tree_samples_storage);
      std::vector<DataSample *> tree_samples(tree_samples_storage.size());
      for(unsigned int s=0; s< tree_samples_storage.size(); ++s){
        tree_samples[s] = &(tree_samples_storage[s]);
      }
      std::sort(tree_samples.begin(), tree_samples.end(), DataSample::DataSamplePointerSort);
      ComputeNewWeights(tree_samples);

      //Train and store
      m_ensemble[t] = tree_train.TrainTree(data_images, samples, tree_samples);
    }else{
      std::vector<DataSample *> tree_samples;
      SubSampleTrainingdata(samples, tree_samples);
      std::sort(tree_samples.begin(), tree_samples.end(), DataSample::DataSamplePointerSort);

      //Train and store
      m_ensemble[t] = tree_train.TrainTree(data_images, samples, tree_samples);
    }

    //  The tree has been trained. We need to update the histograms in the leaf nodes in order to get meaningfull distributions.
    //For this we subsample again with an semi-equal distribution and more samples.
    std::vector<DataSample> histogram_training_samples;
    GetSamples(histogram_training_samples, data_images, m_patch_radius, true, class_weights, true);
    std::sort(histogram_training_samples.begin(), histogram_training_samples.end(), DataSample::DataSampleSort);
    m_ensemble[t]->TrainHistograms(data_images, histogram_training_samples, m_repass_weight);

  //  m_ensemble[t]->CollectAverageLeafPatches(data_images, histogram_training_samples, m_leaf_patch_radius);
  //  m_ensemble[t]->SaveAllLeafPatch(m_patch_image_folder, t);

  }
}

void Forest::getAnnotation(const Utils::DataImage &data_image, cv::Mat* result_pointer, bool parallelize_internal) const{

  const int width = data_image.Width();
  const int height= data_image.Height();
  cv::Mat temp = cv::Mat::zeros(height, width, CV_32FC(m_class_count));
  const cv::Mat & validity_mask = data_image.GetValidityMask();
  *result_pointer = cv::Mat(result_pointer->rows, result_pointer->cols, CV_16SC1, -1.0f);
  const double inv_tree_count = 1.0 / m_tree_count;

  //Initialize the trees
  for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
    (*itr)->InitEvaluation(data_image);
  }

  //For each pixel
#pragma omp parallel for if(parallelize_internal)
  for(int y=0; y<height; ++y ){
    const bool* valid = validity_mask.ptr<bool>(y);
    float* temp_ptr = temp.ptr<float>(y);
    signed char* result_ptr = result_pointer->ptr<signed char>(y);
    for(int x=0; x<width; ++x, ++valid, ++result_ptr){
      //Check if this point is valid!
      if(!(*valid)){
        continue;
      }

      //For each tree
      for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
        //Get the histogram
        const std::vector<float>& hist = (*itr)->GetHistogram(x,y);
        //Add it to the current histogram
        for(unsigned int b=0; b<m_class_count; ++b){
          //temp(x,y,b) *= ((hist[b]+ 1.0f) / (1.0f+ 12.0f* 1.0f));
          temp_ptr[b] += hist[b];
        }
      }
      //Devide in order to create the average.
      double max_prob=0;
      signed char max_index=-1;
      for(unsigned int b=0; b<m_class_count; ++b, ++temp_ptr){
        double current_prob = *temp_ptr * inv_tree_count;
        // current_prob = temp(x,y,b);
        if(current_prob > max_prob){
          max_prob = current_prob;
          max_index=b;
        }
      }
      if(max_index == -1){
        std::cout << "what?" << std::endl;
      }
      *result_ptr= max_index;
    }
  }
}

void Forest::GetUnaryPotential(const Utils::DataImage &data_image, cv::Mat *result, bool parallelize_internal) const{
  const int width = data_image.Width();
  const int height= data_image.Height();
  const double inv_tree_count = 1.0 / m_tree_count;
  const cv::Mat & validity_mask = data_image.GetValidityMask();
  //result->Fill(1.0f/static_cast<float>(m_class_count)); //<-- WTF :D ?
  *result = cv::Mat(height, width, CV_32FC(m_class_count));
  //result.setTo((1.0f/static_cast<float>(m_class_count)));
  result->setTo(0);
  //For each pixel

  //Initialize the trees
  for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
    (*itr)->InitEvaluation(data_image);
  }
#pragma omp parallel for if(parallelize_internal) schedule(dynamic, 1) //<- not really clear if dynamic is better. It seems to be slightly faster, but also more random.
  for(int y=0; y<height; ++y ){
    const bool* valid = validity_mask.ptr<bool>(y);
    float* unary = result->ptr<float>(y);
    for(int x=0; x<width; ++x, ++valid, unary+=m_class_count){
      //Check if this point is valid!
      if(!(*valid)){
        continue;
      }

      //For each tree
      for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
        //Get the histogram
        const std::vector<float>& hist = (*itr)->GetHistogram(x,y);
        //Add it to the
        for(unsigned int b=0; b<m_class_count; b++){
          unary[b] += hist[b];
        }
      }
      //Devide in order to create the average.
      for(unsigned int b=0; b<m_class_count; b++){
        unary[b] = -log(unary[b] *inv_tree_count);
        if(unary[b] > 1000){
          unary[b]  = 1000;
        }
      }
    }
  }
}


void Forest::GetClassHistogramImage(const Utils::DataImage& data_image, cv::Mat* result_pointer, const cv::Mat& valid_points, bool parallelize_internal) const{
  const int width = data_image.Width();
  const int height= data_image.Height();
  const double inv_tree_count = 1.0 / m_tree_count;

  //Initialize the trees
  for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
    (*itr)->InitEvaluation(data_image);
  }
#pragma omp parallel for if(parallelize_internal) schedule(dynamic, 1) //<- not really clear if dynamic is better. It seems to be slightly faster, but also more random.
  //For each pixel
  for(int y=0; y<height; ++y ){
    float* res_ptr = result_pointer->ptr<float>(y);
    const int* valid_ptr = valid_points.ptr<int>(y);
    for(int x=0; x<width; ++x, ++valid_ptr){
      if(*valid_ptr==-1){
        res_ptr+= m_class_count; // First increment the pointer as we are skipping this. 
        continue; //We don't care about this point, as it is never used!
      }
      //For each tree
      for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
        //Get the histogram
        const std::vector<float>& hist = (*itr)->GetHistogram(x,y);
        //Add it to the
        for(unsigned int b=0; b<m_class_count; ++b){
          res_ptr[b] += hist[b];
        }
      }
      //Devide in order to create the average.
      for(unsigned int b=0; b<m_class_count; ++b, ++res_ptr){
        (*res_ptr) *= inv_tree_count;
      }
    }
  }
}

void Forest::GetClassHistogramImage(const Utils::DataImage &data_image, cv::Mat* result_pointer, bool parallelize_internal) const{
  const int width = data_image.Width();
  const int height= data_image.Height();
  const double inv_tree_count = 1.0 / m_tree_count;

  //Initialize the trees
  for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
    (*itr)->InitEvaluation(data_image);
  }
#pragma omp parallel for if(parallelize_internal)
  //For each pixel
  for(int y=0; y<height; ++y ){
    float* res_ptr = result_pointer->ptr<float>(y);
    for(int x=0; x<width; ++x ){
      //For each tree
      for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
        //Get the histogram
        const std::vector<float>& hist = (*itr)->GetHistogram(x,y);
        //Add it to the
        for(unsigned int b=0; b<m_class_count; b++){
          res_ptr[b] += hist[b];
        }
      }
      //Devide in order to create the average.
      for(unsigned int b=0; b<m_class_count; ++b, ++res_ptr){
        (*res_ptr) *= inv_tree_count;
      }
    }
  }
}

int Forest::GetFeatureRequirements() const{
  std::vector<int> features;
  if(m_is_loaded_from_file){
    //already trained, parse the features from the tree.
    for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
      (*itr)->ParseFeatures(features);
    }
  }else{
    //Yet to train, load based on config file.
    Utils::Configuration configuration(m_config_filename);
    bool depth_no_depth = configuration.read<bool>("forest_use_raw");
    features = Utils::Configuration::ParseIntVectorFromString(configuration.read<std::string>("forest_feature_pool"));
    if(depth_no_depth){
      std::vector<int> no_depth = Utils::Configuration::ParseIntVectorFromString(configuration.read<std::string>("forest_no_depth_pool"));
      features.insert(features.end(), no_depth.begin(), no_depth.end());
    }
  }
  int requirements =0;
  for(unsigned int i=0; i < features.size(); i++){
    switch(features[i]){
    case PIXEL_COLOR:
      requirements |= Utils::LAB;
      break;
    case COLOR_ADD:
      requirements |= Utils::LAB;
      break;
    case COLOR_SUB:
      requirements |= Utils::LAB;
      break;
    case COLOR_SUB_ABS:
      requirements |= Utils::LAB;
      break;
    case DEPTH_FEAT:
      requirements |= Utils::DEPTH;
      break;
    case HEIGHT:
      requirements |= Utils::DEPTH | Utils::ACCELEROMETER;
      break;
    case HYBRID_SUB:
      requirements |= Utils::LAB | Utils::DEPTH;
      break;
    case HYBRID_SUB_ABS:
      requirements |= Utils::LAB | Utils::DEPTH;
      break;
    case HYBRID_ADD:
      requirements |= Utils::LAB | Utils::DEPTH;
      break;
    case RELATIVE_DEPTH:
      requirements |= Utils::DEPTH;
      break;
    case COLOR_GRADIENT:
      requirements |= Utils::GRADIENT_COLOR | Utils::LAB;
      break;
    case COLOR_GRADIENT_PATCH:
      requirements |= Utils::GRADIENT_COLOR | Utils::LAB;
      break;
    case GEOMETRICAL:
      requirements |= Utils::GEOMETRIC_FEAT | Utils::DEPTH;
      break;
    case COLOR_GRADIENT_PATCH_COMPARISON:
      requirements |= Utils::GRADIENT_COLOR | Utils::LAB;
      break;
    case COLOR_GRADIENT_PATCH_SCALED:
      requirements |= Utils::GRADIENT_COLOR | Utils::LAB | Utils::DEPTH;
      break;
    case DEPTH_GRADIENT_PATCH:
      requirements |= Utils::GRADIENT_DEPTH | Utils::DEPTH;
      break;
    case DEPTH_GRADIENT_PATCH_SCALED:
      requirements |= Utils::GRADIENT_DEPTH | Utils::DEPTH;
      break;
    case COLOR_PATCH:
      requirements |= Utils::DEPTH | Utils::LAB | Utils::LAB_INTEGRAL;
      break;
    case COLOR_GRADIENT_PATCH_COMPARISON_SCALED:
      requirements |= Utils::GRADIENT_COLOR | Utils::LAB | Utils::DEPTH;
      break;
    case NORMAL_FEATURE:
      requirements |=  Utils::DEPTH | Utils::ACCELEROMETER | Utils::NORMALS;
      break;
    case ORDINAL_DEPTH:
      requirements |=  Utils::DEPTH;
      break;
    case X_PIXEL_POS:
      break;
    case Y_PIXEL_POS:
      break;
    case DEPTH_NAN_CHECK:
      break;
    default:
      std::cerr << "Feature type " << features[i] << " does not exist! Maybe you forgot to add it in the 'forest.cc' file?" << std::endl;
      throw std::runtime_error("Wrong features required!");
      break;
    }
  }
  return requirements;
}


void Forest::Save(const std::string& config_filename) {
  //Get filename for the forests
  Utils::Configuration conf(config_filename);
  std::string forest_filename = conf.read<std::string>("forest_filename");
  SaveByFilename(forest_filename);

}
void Forest::SaveByFilename(const std::string& forest_filename){
  // Serialize
  m_is_loaded_from_file = true;
  std::ofstream ofs(forest_filename.c_str());
  {
    boost::archive::text_oarchive oa(ofs);
    oa << *this;
  }
}

Forest *Forest::Load(const std::string &config_filename){
  //Get filename for the forests
  Utils::Configuration conf(config_filename);
  std::string forest_filename = conf.read<std::string>("forest_filename");
  return LoadByFilename(forest_filename);
}

Forest* Forest::LoadByFilename(const std::string& forest_filename){
  // Deserialize
  Forest *forest = new Forest();
  {
    std::ifstream ifs(forest_filename.c_str());
    boost::archive::text_iarchive ia(ifs);
    ia >> *forest;
  }
  return (forest);
}

void Forest::PrintForest() const{
  std::cout << "\nForest with the following " << m_ensemble.size() << " trees:" << std::endl;
//    for(std::vector<Tree *>::const_iterator itr= m_ensemble.begin(); itr != m_ensemble.end(); ++itr){
//      std::cout << (*itr)->PrintTree();
//      (*itr)->PrintLeafNodeDistributions();
//    }
}

void Forest::TrainInParallel(bool flag){
  m_train_trees_parallel = flag;
}


void Forest::GetSamples(std::vector<DataSample> &data_samples, const std::vector<Utils::DataImage> &data_images, unsigned int patch_radius, bool store_classweights, std::vector<double> &class_weights, bool equal_sampling){

  std::vector<unsigned int> considered_images;

  // Random across imags type [0..(n-1)]
  boost::uniform_int<unsigned int> uniform_dist(0, data_images.size()-1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_image(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);


  //Allow all images
  for(unsigned int i=0; i < data_images.size(); ++i){
    considered_images.push_back(i);
  }

  //Only pick a subset.
  //  for(int i=0; i < data_images.size()/m_num_samples; ++i){
  //    considered_images.push_back(rgen_image());
  //  }

  //Resize the vector to the right capacity
  int max=-1;
  if(equal_sampling){
    data_samples.resize(m_num_samples*4);
    max = data_samples.size()/m_class_count +1;
  }else{
    data_samples.resize(m_num_samples);
  }
  
  // Random across possible images
  boost::uniform_int<unsigned int> uniform_dist2(0, considered_images.size()-1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_images(Utils::RandomSource::GlobalRandomSampler(), uniform_dist2);
  
  // Sample a random (x,y) coordinate for the image. (Assuming all pixels are of equal size.

  //Here, be sure to take care of the patch radius. The patches should not be affected by broken borders.
  boost::uniform_int<unsigned int> uniform_x(patch_radius, data_images[0].Width()- 1 - patch_radius);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_x(Utils::RandomSource::GlobalRandomSampler(), uniform_x);
  boost::uniform_int<unsigned int> uniform_y(patch_radius, data_images[0].Height() - 1 - patch_radius);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_y(Utils::RandomSource::GlobalRandomSampler(), uniform_y);


  //boost::uniform_real<float> uniform_reject(0,1.0f/static_cast<float>(m_class_count));
  //boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rgen_reject(Utils::RandomSource::GlobalRandomSampler(), uniform_reject);

  int image_idx, x, y;
  //int failed_counter = 0;
  signed char label = -1;
  std::vector<float> class_count(m_class_count, 0);
  //Loop over the images and extract the samples
  for(unsigned int i=0; i < data_samples.size(); ++i){
    image_idx = considered_images[rgen_images()];
    x = rgen_x();
    y = rgen_y();
    label = data_images[image_idx].GetAnnotation().at<signed char>(y,x);

    if ( label < 0) {
      i--;
      continue;
    }
    if(equal_sampling){
      //   std::cout << "count, max:  " << class_count[label] << " " << max << std::endl;
      //Check the class distribution.
      if( class_count[label]  <= max ){
        //The class count is small, add it, do not reject it.
        DataSample sample = {image_idx, static_cast<unsigned short>(x), static_cast<unsigned short>(y), label,1.0, label};
        data_samples[i] = sample;
        class_count[label] = 1 + class_count[label];
      }else{
        // The class count is rather big already. We reject it and go on.
        i--;
        //        failed_counter++;
        //        if(failed_counter > 10e6){
        //          throw std::runtime_error("Error: Equally sampling totally failed! Ending...");
        //        }
      }
    }else{
      // Append to sample list
      DataSample sample = {image_idx, static_cast<unsigned short>(x), static_cast<unsigned short>(y), label,1.0, label};
      data_samples[i] = sample;

    }
  }

  //Check if we want to want to store the class weights as well.
  // See Section 2 in [Shotton2008].
  if (store_classweights) {
    double total = 0.0;
    std::map<signed char, double> class_count;
    for (unsigned int n = 0; n < data_samples.size(); ++n) {
      class_count[data_samples[n].center_label] += 1.0;
      total += 1.0;
    }
    std::cout << "Class-rebalancing data, original relative proportions:" << std::endl;
    for (std::map<signed char, double>::const_iterator ci = class_count.begin(); ci != class_count.end(); ++ci) {
      std::cout << "  class " << static_cast<int>(ci->first) << " has " << (100.0 * (ci->second / total)) << " percent" << std::endl;
    }

    class_weights.resize(m_class_count);
    for(unsigned int i=0; i < m_class_count; ++i){
      class_weights[i] = 1.0 / class_count[i];
    }

    for (unsigned int n = 0; n < data_samples.size(); ++n) {
      signed char class_id = data_samples[n].center_label;
      data_samples[n].label_weight = class_weights[class_id];
    }
  }
}

void Forest::SubSampleTrainingdata(std::vector<DataSample> &training_samples, std::vector<DataSample *> &tree_samples){
  //Random across imags type [0..(n-1)]
  boost::uniform_int<unsigned int> uniform_dist(0, training_samples.size()-1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_sample(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);
  tree_samples.resize(training_samples.size()/m_sub_sample_per_tree);
  for(unsigned int i=0 ; i <tree_samples.size(); ++i){
    tree_samples[i] = &(training_samples[rgen_sample()]);
  }
}

void Forest::SubSampleTrainingdata(std::vector<DataSample> &training_samples, std::vector<DataSample> &tree_samples){
  //Random across imags type [0..(n-1)]
  boost::uniform_int<unsigned int> uniform_dist(0, training_samples.size()-1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_sample(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);
  tree_samples.resize(training_samples.size()/m_sub_sample_per_tree);
  for(unsigned int i=0 ; i <tree_samples.size(); ++i){
    tree_samples[i] = training_samples[rgen_sample()];
  }
}

void Forest::ComputeNewWeights(std::vector<DataSample *> &data_samples){
  double total = 0.0;
  std::map<signed char, double> class_count;
  for (unsigned int n = 0; n < data_samples.size(); ++n) {
    class_count[data_samples[n]->center_label] += 1.0;
    total += 1.0;
  }
  std::cout << "Weights after subsampling the training data for this tree:" << std::endl;
  for (std::map<signed char, double>::const_iterator ci = class_count.begin(); ci != class_count.end(); ++ci) {
    std::cout << "  class " << static_cast<int>(ci->first) << " has " << (100.0 * (ci->second / total)) << " percent" << std::endl;
  }

  for (unsigned int n = 0; n < data_samples.size(); ++n) {
    signed char class_id = data_samples[n]->center_label;
    data_samples[n]->label_weight = 1.0 / class_count[class_id];

  }
}

bool Forest::SortData(const DataSample &i, const DataSample &j) {
  return i.image_id < j.image_id;
}






