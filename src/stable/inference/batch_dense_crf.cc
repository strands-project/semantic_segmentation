/**
 * @file   batch_dense_crf.cc
 * @author Alexander Hermans
 * @date   Mon Oct 29 2012 4:30pm
 *
 * @brief  CRF evaluation application. Loads all test files and evaluates the crf.
 */
#include <cstdio>
#include <string>
#include <cmath>

//Inference includes
#include <inference/densecrf.hh>
#include <inference/densecrf_evaluator.hh>

//Util includes
#include <utils/data_image.hh>
#include <utils/time_stamp.hh>
#include <utils/static_dataset.hh>
#include <utils/configuration.hh>

// OpenCV includes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Boost includes
#include <boost/progress.hpp>

// TBB includes
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>

#define CPUS  8

class ParallelEvaluator {
  Utils::StaticDataset * m_data;
  int m_dataset_type;
  Inference::DenseCRFEvaluator m_evaluator;
  boost::progress_display* m_progress;
  int* m_copy_counter;

  float* m_image_counter;
  float* m_load_counter;
  float* m_inference_counter;
  float* m_total_counter;
  tbb::mutex* m_mutex;

public:
  /// Returns the image count, just to start the evaluator in the right way.
  int GetImageCount(){
    return m_data->ImageCount();
  }


  ParallelEvaluator(const char * config_filename):
    m_evaluator(config_filename){
    //Setup the dataset
    Utils::Configuration configuration(config_filename);
    m_dataset_type = configuration.read<int>("dataset_type");
    int requirements = m_evaluator.GetLoadRequirements();
    m_data = Utils::StaticDataset::LoadDataset(configuration, Utils::StaticDataset::TEST, requirements);

    //Stops the evaluator from spawning new threads. This would slow down the evaluation when we are parallelizing on an image level.
    m_evaluator.StopParallelization();

    //Needed to fix the TBB calling the destructor several times...
    //So... This is a smart pointer, I just don't want to include boost for this!
    m_copy_counter = new int[1];
    (*m_copy_counter) =1;
    m_image_counter = new float[1];
    (*m_image_counter) =0;
    m_load_counter = new float[1];
    (*m_load_counter) = 0;
    m_inference_counter = new float[1];
    (*m_inference_counter) = 0;
    m_total_counter = new float[1];
    (*m_total_counter) = 0;
    m_mutex = new tbb::mutex;
    m_progress = new boost::progress_display(m_data->ImageCount());

  }

  ParallelEvaluator(const ParallelEvaluator& other){
    m_data = other.m_data;
    m_evaluator = other.m_evaluator;
    m_dataset_type = other.m_dataset_type;
    m_copy_counter = other.m_copy_counter;
    m_progress = other.m_progress;
    m_image_counter = other.m_image_counter;
    m_load_counter = other.m_load_counter;
    m_inference_counter = other.m_inference_counter;
    m_total_counter = other.m_total_counter;
    m_mutex = other.m_mutex;
    m_mutex->lock();
    (*m_copy_counter)++;
    m_mutex->unlock();
  }

  ~ParallelEvaluator(){
    m_mutex->lock();
    (*m_copy_counter)--;
    m_mutex->unlock();
    if((*m_copy_counter) == 0){
      delete m_data;
      delete m_copy_counter;
      std::cout << "------------------------------------------" << std::endl;
      std::cout << "Inference statistics " << std::endl;
      std::cout << "Average time for loading:    " << *m_load_counter /(CPUS* *m_image_counter) << std::endl;
      std::cout << "Average time for inference:  " << *m_inference_counter /(CPUS* *m_image_counter) << std::endl;
      std::cout << "Average time total :         " << *m_total_counter /(CPUS* *m_image_counter) << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      delete m_image_counter;
      delete m_load_counter;
      delete m_inference_counter;
      delete m_total_counter;
      delete m_progress;
      delete m_mutex;
    }
  }



  void  operator() (const tbb::blocked_range<int>& r) const{
    for ( int i = r.begin(); i != r.end(); i++ ) {
      Utils::TimeStamp t;
      t.Stamp();
      //Load the image
      Utils::DataImage current = m_data->GenerateImage(i);
      m_mutex->lock();
      (*m_image_counter) +=1;
      m_mutex->unlock();
      cv::Mat segmentation_result(current.Height(), current.Width(), CV_8SC1);
      m_mutex->lock();
      (*m_load_counter) +=t.Elapsed();
      m_mutex->unlock();
      t.Stamp();
      //Do the actual inference
      m_evaluator.Evaluate(current, segmentation_result);
      m_mutex->lock();
      (*m_inference_counter) += t.Elapsed();
      m_mutex->unlock();
      t.Stamp();
      // Store the result
      cv::Mat result = m_data->GetColorCoding()->LabelToBgr(segmentation_result);
      cv::imwrite(current.GetResultFilename(), result);
      m_mutex->lock();
      (*m_total_counter) += t.TotalElapsed();
      ++(*m_progress);
      m_mutex->unlock();
    }
  }
};






int main( int argc, char* argv[]){
  if (argc != 2) {
    std::cerr << "Usage: " << std::endl;
    std::cerr << "test_crf <config file> " << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "(test) Evaluating the dense crf" << std::endl;

  ParallelEvaluator eval(argv[1]);
  tbb::parallel_for(tbb::blocked_range<int>(0, eval.GetImageCount(), eval.GetImageCount()/CPUS+1),eval);


}
