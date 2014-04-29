#ifndef GENERATOR_SPLIT_FUNCTION_HH
#define GENERATOR_SPLIT_FUNCTION_HH

//Local includes
#include "features/tree_feature.hh"
#include "features/height_feature.hh"
#include "features/color_feature_addition.hh"
#include "features/color_feature_subtraction.hh"
#include "features/color_feature_subtraction_abs.hh"
#include "features/single_pixel_color_feature.hh"
#include "features/image_coordinate_feature.hh"
#include "features/depth_comparison_feature.hh"
#include "features/depth_color_feature_addition.hh"
#include "features/depth_color_feature_subtraction.hh"
#include "features/depth_color_feature_subtraction_abs.hh"
#include "features/relative_depth_feature.hh"
#include "features/color_gradient_pixel_feature.hh"
#include "features/color_gradient_patch_feature.hh"
#include "features/geometrical_feature.hh"
#include "features/color_gradient_patch_comparison_feature.hh"
#include "features/depth_check_feature.hh"
#include "features/depth_gradient_patch_feature.hh"
#include "features/color_patch_feature.hh"
#include "features/normal_feature.hh"
#include "features/ordinal_depth_feature.hh"


// Utils includes
#include <utils/random_source.hh>
#include <utils/console_tree.hh>
#include <utils/data_image.hh>

// STL includes
#include <limits>

// TBB includes
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace Rdfs{

template <class ImageT>
class SplitFunctionGenerator
{
public:
  SplitFunctionGenerator(){
  }


  SplitFunctionGenerator(const std::vector<ImageT>* image_pointers, unsigned int num_splits, unsigned int num_thresholds, unsigned int num_classes):
     m_image_pointers(image_pointers), m_split_count(num_splits), m_thresholds_count(num_thresholds), m_class_count(num_classes){

  }



  TreeFeature *GetSplitFunction(const std::vector<DataSample *> &samples, std::vector<DataSample *> &left_samples, std::vector<DataSample *> &right_samples, int patch_radius, unsigned int max_depth_check_distance, std::vector<unsigned int> gradient_parameters, Utils::Console_Tree* console_tree, std::vector<int> features, float depth){

    boost::uniform_int<int> uniform_dist_x(0, samples.size()-1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_index(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_x);

    //Setup arrays to store the results
    std::vector<float> f_value(samples.size());
    std::vector<float> current_thresholds(m_thresholds_count);
    TreeFeature *feature_current;
    float f_min;
    float f_max;
    bool ok_feature;

    //this stores the results for the best feature
    std::vector<float> f_value_best(samples.size());
    TreeFeature *feature_best = NULL;
    float threshold_best = 0;
    float score_best = -std::numeric_limits<float>::infinity();

    //Counter to make sure we do not get stuck during training!
    int failed_tries =0;

    //For splitcount
    for(int i=0; i< m_split_count; ++i){
      //sample feature space
      feature_current = GetRandomFeature(patch_radius, max_depth_check_distance, gradient_parameters, features);
      //Evaluate and store min max
      f_min = std::numeric_limits<float>::infinity();
      f_max = -std::numeric_limits<float>::infinity();


      for(int j=0; j < samples.size(); ++j){
        DataSample current = *(samples[j]);
        float fval= feature_current->ExtractFeature(current.x_pos, current.y_pos, (*m_image_pointers)[current.image_id]);
        if(!isnan(fval)){
          f_min = std::min(f_min, fval);
          f_max = std::max(f_max, fval);
        }
        f_value[j]  = fval;
      }

      ok_feature = false;
      if (fabs(f_max) > 0.0 && ((f_max - f_min) / fabs(f_max)) > 1e-5) {
        ok_feature = true;
      } else if ((f_max - f_min) > 1e-8) {
        ok_feature = true;
      }
      // Delete feature if not valid
      if (ok_feature==false) {
        delete (feature_current);
        feature_current = NULL;
        i--; //Recalculate for a new one.
        failed_tries++;
        if(failed_tries > 500){
          //We have tried to compute a good feature for too many times.
          //This sample set is not feasible.
          throw std::runtime_error("Too many features were bad features, either the sample set is bad, or the selected features are not good.");
        }
        continue;
      }
      //printf("%c[%d;%dH",27,0,40);   std::cout << feature_current->GetFeatureType() << "  "  << f_min << " , " << f_max << "                " << std::endl;
      GetRandomThreshold(f_min, f_max, current_thresholds, feature_current->GetFeatureType());
      //for threshold count, check the bist splits based on score.
      for(int k=0; k < m_thresholds_count; ++k){
        float score = Score(f_value, current_thresholds[k], samples, depth);
        if(score > score_best){
          //We found a better feature/threshold combination
          score_best = score;
          //If the better feature is a different feature than the currently best
          if(feature_best != feature_current ){
            //Delete the old best feature and reset it with the new one.
            if(feature_best != NULL){
              delete(feature_best);
            }
            feature_best = feature_current;
            f_value_best = f_value; // TODO: Check if this is slowing down the system a lot!
          }
          threshold_best = current_thresholds[k];
        } //Otherwise the feature is not better. We do not do anything.

      }
      //After the loop, we need to check if the current feature is actually the best feature. If not we delete it.
      if(feature_best != feature_current){
        delete(feature_current);
      }
      console_tree->Update(static_cast<int>((i*100/m_split_count)));
    }
    //Split up the data into the input vectors
    for(int j=0; j < samples.size(); ++j){
      //if(f_value_best[j] <= threshold_best){
      DataSample * current = samples[j];
      if(feature_best->ExtractFeature(current->x_pos, current->y_pos, (*m_image_pointers)[current->image_id])  <= threshold_best){
        left_samples.push_back(current);
      }else{
        right_samples.push_back(current);
      }
    }


    //Return the best spliting function. Create it newly here
    feature_best->SetThreshold(threshold_best);
    return feature_best;
    //return new SplitFunction( feature_best, threshold_best);
  }




private:


/*

  float Score(std::vector<float> &f_values, float thresh, const std::vector<DataSample *> &samples, float depth) const {
    float total = samples.size();
    float total_left = 0;
    float total_right = 0;
    std::vector<float> left_class_count(m_class_count, 0);
    float total_weighted_left=0;
    std::vector<float> right_class_count(m_class_count, 0);
    float total_weighted_right=0;


    for(int i=0; i< total; ++i){
      float current_weight = samples[i]->label_weight;
      if(f_values[i] <= thresh){ //left
        total_left ++;
        left_class_count[samples[i]->label] += current_weight;
        total_weighted_left += current_weight;
      }else{ //right
        total_right ++;
        right_class_count[samples[i]->label] += current_weight;
        total_weighted_right += current_weight;
      }
    }

    float e_left, e_right = 0;
    for(int l=0; l < m_class_count; ++l){
      float p;
      p = left_class_count[l] / total_weighted_left;
      if(p!=0){
        e_left  += p * log(p);
      }
      p = right_class_count[l] / total_weighted_right;
      if(p!=0){
        e_right += p * log(p);
      }
    }

    return ( total_left*e_left + total_right*e_right)/total ;
  }

*/


  /**
     * @brief Computes the normalized information gain score of a split
     *
     * @param f_values Results from the samples for a given feature
     * @param thresh   Current threshold
     * @param samples  Pointers to the current samples
     * @param depth    Depth of current split, used for depth sensitive scores
     *
     * @return Socre of the split
  */
  float Score(std::vector<float> &f_values, float thresh, const std::vector<DataSample *> &samples, float depth) const {

    // [class] equals count in the respective branch
    std::map<signed char, float> all_class;
    std::map<signed char, float> left_class;
    std::map<signed char, float> right_class;
    float total = 0.0;
    float total_left = 0.0;
    float total_right = 0.0;

    // Make one linear pass through all samples of this node and obtain the total and class conditional counts.
    int counter=0;
    for (std::vector<DataSample *>::const_iterator si = samples.begin(); si != samples.end(); ++si) {
      signed char class_label = (*si)->label;
      bool leftright = f_values[counter] <= thresh;

      double weight = (*si)->label_weight;
      assert(weight >= 0.0);
      all_class[class_label] += weight;
      if (leftright) {
        left_class[class_label] += weight;
        total_left += weight;
      } else {
        right_class[class_label] += weight;
        total_right += weight;
      }
      total += weight;
      counter++;
    }

    // H_s: "split entropy", i.e. a measure of the split balancedness
    float H_s = (- total_left * log2(total_left) - total_right * log2(total_right) + total * log2(total)) / total;

    // H_c: classification entropy over class-distribution in current set
    float H_c = 0.0;
    for (std::map<signed char, float>::const_iterator ci = all_class.begin(); ci != all_class.end(); ++ci) {
      H_c -= ci->second * log2(ci->second);
    }
    H_c += total * log2(total);
    H_c /= total;

    // Compute information gain due to split decision
    float I_split = 0.0;
    // Empirical p(B = left), p(B = right)
    float p_l = total_left / total;
    float p_r = total_right / total;
    for (std::map<signed char, float>::const_iterator ci = left_class.begin(); ci != left_class.end(); ++ci) {
      // Empirical p(B = left, C = ci->first)
      float p_lc = ci->second / total;
      float p_c = all_class[ci->first] / total;

      assert(p_c > 0.0);
      I_split += p_lc * log2(p_lc / (p_l * p_c));
    }
    for (std::map<signed char, float>::const_iterator ci = right_class.begin(); ci != right_class.end(); ++ci) {
      // Empirical p(B = right, C = ci->first)
      float p_rc = ci->second / total;
      float p_c = all_class[ci->first] / total;

      assert(p_c > 0.0);
      I_split += p_rc * log2(p_rc / (p_r * p_c));
    }

    return ((2.0 * I_split) / (H_s + H_c));
  }

  TreeFeature *GetRandomFeature(int patch_radius, int max_depth_check_distance,  std::vector<unsigned int> gradient_parameters, std::vector<int> features){

    boost::uniform_int<unsigned int> uniform_dist(0, features.size()-1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<unsigned int> > rgen_ft(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);

    // take a random feature from the selected features.
    unsigned int feature_type = features.at(rgen_ft());



    int x1,x2, y1, y2, c1, c2;

    if(feature_type < 6){ // color features
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-patch_radius, patch_radius);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);

      // Color channel [0-2]
      boost::uniform_int<int> uniform_dist_cc(0, 2);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_cc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_cc);


      x1 = rgen_xy();
      y1 = rgen_xy();

      // Second, disjoint pixel
      do {
        x2 = rgen_xy();
        y2 = rgen_xy();
      } while (x1==x2 && y1==y2);
      c1 = rgen_cc();
      c2 = rgen_cc();
      assert(c1 >= 0 && c1 < 3);
      assert(c2 >= 0 && c2 < 3);
    }

    if(feature_type==PIXEL_COLOR){
      return new ColorFeature(x1, y1, c1);
    }

    if(feature_type==COLOR_ADD){
      return new ColorFeatureAddition(x1, y1, c1, x2, y2, c2);
    }

    if(feature_type==COLOR_SUB){
      return new ColorFeatureSubtraction(x1, y1, c1, x2, y2, c2);
    }

    if(feature_type==COLOR_SUB_ABS){
      return new ColorFeatureSubtractionAbs(x1, y1, c1, x2, y2, c2);
    }

    if(feature_type==X_PIXEL_POS){
      return new ImageCoordinateFeature(X_PIXEL_POS);
    }

    if(feature_type==Y_PIXEL_POS){
      return new ImageCoordinateFeature(Y_PIXEL_POS);
    }


    if(feature_type==HEIGHT){
      return new HeightFeature();
    }

    if(feature_type==RELATIVE_DEPTH){
      boost::uniform_int<int> uniform_dist_fc(0, 1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_fc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_fc);

      return new RelativeDepthFeature(rgen_fc());
    }

    if(feature_type==DEPTH_FEAT || feature_type==HYBRID_SUB || feature_type==HYBRID_SUB_ABS || feature_type==HYBRID_ADD){ //depth features
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-max_depth_check_distance, max_depth_check_distance);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);
      x1 = rgen_xy();
      y1 = rgen_xy();

      // Second, disjoint pixel
      do {
        x2 = rgen_xy();
        y2 = rgen_xy();
      } while (x1==x2 && y1==y2);


      // Color channel [0-2]
      boost::uniform_int<int> uniform_dist_cc(0, 2);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_cc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_cc);

     // boost::uniform_int<int> uniform_dist_cc2(0, 3);
     // boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_cc2(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_cc2);



      c1 = rgen_cc();
      c2 = rgen_cc();

      if(feature_type==DEPTH_FEAT){
        return new DepthComparisonFeature(x1, y1, x2, y2);
      }

      if(feature_type==HYBRID_SUB){
        return new DepthColorFeatureSubtraction(x1, y1, c1, x2, y2, c2);
      }
      if(feature_type==HYBRID_SUB_ABS){
        return new DepthColorFeatureSubtractionAbs(x1, y1, c1, x2, y2, c2);
      }

      if(feature_type==HYBRID_ADD){
        return new DepthColorFeatureAddition(x1, y1, c1, x2, y2, c2);
      }

    }


    if(feature_type==COLOR_GRADIENT){ //gradient feature simply comparing 2 pixels in a gradient patch centered around the image.

      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-patch_radius, patch_radius);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);
      x1 = rgen_xy();
      y1 = rgen_xy();

      // Second, disjoint pixel
      do {
        x2 = rgen_xy();
        y2 = rgen_xy();
      } while (x1==x2 && y1==y2);

      //gradient channels 0,through max bins -1.
      boost::uniform_int<int> uniform_dist_gradientc(0, gradient_parameters[0]-1);

      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_gradientc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_gradientc);
      c1= rgen_gradientc();
      c2= rgen_gradientc();

      return new ColorGradientPixelFeature(x1, y1, c1, x2, y2, c2);

    }

    if(feature_type==COLOR_GRADIENT_PATCH || feature_type==COLOR_GRADIENT_PATCH_SCALED || feature_type==DEPTH_GRADIENT_PATCH || feature_type==DEPTH_GRADIENT_PATCH_SCALED){ //gradient feature taking a gradient patch with a certain offset from the pixel
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-gradient_parameters[3], gradient_parameters[3]);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);
      x1 = rgen_xy();
      y1 = rgen_xy();

      boost::uniform_int<int> uniform_dist_xysize(gradient_parameters[1],gradient_parameters[2]);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xysize(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xysize);
      x2 = rgen_xysize();
      y2 = rgen_xysize();

      //gradient channels 0,through max bins -1.
      boost::uniform_int<int> uniform_dist_gradientc(0, gradient_parameters[0] -1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_gradientc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_gradientc);
      c1= rgen_gradientc();
      c2= rgen_gradientc();

      if(feature_type==DEPTH_GRADIENT_PATCH || feature_type==DEPTH_GRADIENT_PATCH_SCALED){// DEPTH_GRADIENT_PATCH or DEPTH_GRADIENT_PATCH_SCALED
        return new DepthGradientPatchFeature(x1,y1,c1, x2,y2, feature_type==DEPTH_GRADIENT_PATCH_SCALED);
      }else{ // COLOR_GRADIENT_PATCH or COLOR_GRADIENT_PATCH_SCALED
        return new ColorGradientPatchFeature(x1,y1,c1, x2,y2, feature_type==COLOR_GRADIENT_PATCH_SCALED);
      }
    }

    if(feature_type==GEOMETRICAL){ // 3d feature
      //Select one out of three different feature types.
      boost::uniform_int<int> uniform_dist_type(0,2);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_3d_type(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_type);
      c1= rgen_3d_type();
      return new GeometricalFeature(c1);
    }


    if(feature_type==COLOR_GRADIENT_PATCH_COMPARISON || feature_type==COLOR_GRADIENT_PATCH_COMPARISON_SCALED){ //gradient feature patch comparrison
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-gradient_parameters[3], gradient_parameters[3]);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);
      x1 = rgen_xy();
      y1 = rgen_xy();

      boost::uniform_int<int> uniform_dist_xysize(gradient_parameters[1],gradient_parameters[2]);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xysize(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xysize);
      x2 = rgen_xysize();
      y2 = rgen_xysize();

      //gradient channels 0,through max bins -1.
      boost::uniform_int<int> uniform_dist_gradientc(0, gradient_parameters[0] -1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_gradientc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_gradientc);
      c1= rgen_gradientc();
      c2= rgen_gradientc();


      int x3, y3;
      // Second, disjoint pixel
      do {
        x3 = rgen_xy();
        y3 = rgen_xy();
      } while (x1==x3 && y1==y3);


      int x4 = rgen_xysize();
      int y4 = rgen_xysize();
      return new ColorGradientPatchComparisonFeature(x1, y1,c1, x2, y2, x3, y3, c2, x4, y4, feature_type==20);

    }

    if(feature_type==COLOR_PATCH){ //color_patch
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-gradient_parameters[3], gradient_parameters[3]);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);
      x1 = rgen_xy();
      y1 = rgen_xy();

      boost::uniform_int<int> uniform_dist_xysize(gradient_parameters[1],gradient_parameters[2]);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xysize(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xysize);
      x2 = rgen_xysize();
      y2 = rgen_xysize();

      //Color channels
      boost::uniform_int<int> uniform_dist_gradientc(0,2);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_gradientc(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_gradientc);
      c1= rgen_gradientc();
      c2= rgen_gradientc();


      int x3, y3;
      // Second, disjoint pixel
      do {
        x3 = rgen_xy();
        y3 = rgen_xy();
      } while (x1==x3 && y1==y3);


      int x4 = rgen_xysize();
      int y4 = rgen_xysize();
      return new ColorPatchFeature(x1, y1,c1, x2, y2, x3, y3, c2, x4, y4);
    }

    if(feature_type==NORMAL_FEATURE){ //normals
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-max_depth_check_distance,max_depth_check_distance);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);
      x1 = rgen_xy();
      y1 = rgen_xy();

      // Second, disjoint pixel
      do {
        x2 = rgen_xy();
        y2 = rgen_xy();
      } while (x1==x2 && y1==y2);

      boost::uniform_int<int> uniform_dist_normals(0,4);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_normal_c(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_normals);
      c1= rgen_normal_c();

      //Give the up vector, this should give the option to distinguish wall from floors etc.
      Eigen::Vector3f vec(0.0f, 1.0f, 0.0f);
      return new NormalFeature(x1, y1, x2, y2, c1, vec);
    }


    if(feature_type==ORDINAL_DEPTH){
      // (x,y) offsets
      boost::uniform_int<int> uniform_dist_xy(-max_depth_check_distance, max_depth_check_distance);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_xy(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_xy);

      //TODO unhardcode this.
      unsigned num = 3;

      std::vector<int> x_offset(num);
      std::vector<int> y_offset(num);

      for(unsigned int k = 0; k < num; k++){
        x_offset[k] = rgen_xy();
        y_offset[k] = rgen_xy();
      }
      //Random index
      boost::uniform_int<int> uniform_dist_index(0,num-1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<int> > rgen_random_index(Utils::RandomSource::GlobalRandomSampler(), uniform_dist_index);

      return new OrdinalDepthFeature(rgen_random_index(), x_offset, y_offset);
    }

    //This point should never be reached as it means you forgot to add some code here!
    std::cerr << "Feature with type " << feature_type << " was required! I don't know how to creat this. Try adding it in the split_function_generator.hh" << std::endl;
    throw std::runtime_error("Error: feature is not supported!");
  }


  void GetRandomThreshold(float f_min, float f_max, std::vector<float> &thresholds, int current_feature_type = -1){
    if(!(f_min < f_max)){
      std::cout << "max bigger than min! f_min, f_max, feature_type: " <<f_min <<  ", " << f_max <<  ", " << current_feature_type << std::endl;
    }
    if(f_min==-std::numeric_limits<float>::infinity()){
      std::cout << "min is -infinity! f_min, f_max, feature_type: " <<f_min <<  ", " << f_max <<  ", " << current_feature_type << std::endl;
    }
    if(f_max==std::numeric_limits<float>::infinity()){
      std::cout << "max is infinity! f_min, f_max, feature_type: " <<f_min <<  ", " << f_max <<  ", " << current_feature_type << std::endl;
    }
    if(!(f_min==f_min)){
      std::cout << "min is NaN! f_min, f_max, feature_type: " <<f_min <<  ", " << f_max <<  ", " << current_feature_type << std::endl;
    }
    if(!(f_max==f_max)){
      std::cout << "max is NaN! f_min, f_max, feature_type: " <<f_min <<  ", " << f_max <<  ", " << current_feature_type << std::endl;
    }
    boost::uniform_real<float> uniform_dist(f_min, f_max);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > rgen(Utils::RandomSource::GlobalRandomSampler(), uniform_dist);
    for(unsigned int i=0; i < thresholds.size(); ++i){
      thresholds[i] = rgen();
    }
  }

private:
  const std::vector<ImageT>*                    m_image_pointers;
  unsigned int                                  m_split_count;
  unsigned int                                  m_thresholds_count;
  unsigned int                                  m_class_count;

};
}
#endif // GENERATOR_SPLIT_FUNCTION_HH
