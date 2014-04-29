// local includes
#include "forest.hh"

// Utils includes
#include <utils/configuration.hh>
#include <utils/static_dataset.hh>
#include <utils/time_stamp.hh>
#include <utils/cv_util.hh>

// Boost includes
#include <boost/progress.hpp>

// TBB includes
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#define CPUS  8

class ParallelEvaluator {
  const Utils::StaticDataset * m_dataset;
  std::string m_forest_filename;
  const Rdfs::Forest * m_forest;
  boost::progress_display* m_progress;
  bool m_get_unaries;
  int m_class_count;
  int m_dataset_type;

  int* m_copy_counter;
  float* m_image_counter;
  float* m_load_counter;
  float* m_inference_counter;
  float* m_total_counter;
  tbb::mutex* m_mutex;

public:
  ParallelEvaluator(const Utils::StaticDataset * dataset, const std::string forest_filename,
                    boost::progress_display* progress, bool get_unaries,
                    int class_count, int dataset_type) :
    m_dataset(dataset), m_forest_filename(forest_filename), m_progress(progress), m_get_unaries(get_unaries),
    m_class_count(class_count), m_dataset_type(dataset_type){

    m_forest = Rdfs::Forest::LoadByFilename(forest_filename);

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
  }

  ParallelEvaluator(const ParallelEvaluator& other){
    m_dataset = other.m_dataset;
    m_forest = Rdfs::Forest::Load(other.m_forest_filename);
    m_forest_filename = other.m_forest_filename;
    m_progress = other.m_progress;
    m_get_unaries = other.m_get_unaries;
    m_class_count = other.m_class_count;
    m_dataset_type = other.m_dataset_type;
    m_copy_counter = other.m_copy_counter;
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
    delete m_forest;
    m_mutex->lock();
    (*m_copy_counter)--;
    m_mutex->unlock();
    if((*m_copy_counter) == 0){
      delete m_copy_counter;
      std::cout << "------------------------------------------" << std::endl;
      std::cout << "Inference statistics " << std::endl;
      std::cout << "Average time for loading:    " << *m_load_counter /(CPUS* *m_image_counter) << std::endl;
      std::cout << "Average time for evaluating: " << *m_inference_counter /(CPUS* *m_image_counter) << std::endl;
      std::cout << "Average time total :         " << *m_total_counter /(CPUS* *m_image_counter) << std::endl;
      std::cout << "------------------------------------------" << std::endl;
      delete m_image_counter;
      delete m_load_counter;
      delete m_inference_counter;
      delete m_total_counter;
      delete m_mutex;
    }
    //Everything else is taken care of outside of here.
  }

  void  operator() (const tbb::blocked_range<int>& r) const{
    if(!m_get_unaries){ //Just save the unary result images.
      for ( int i = r.begin(); i != r.end(); i++ ) { // iterates over the entire chunk
        Utils::TimeStamp t;
        m_mutex->lock();
        (*m_image_counter) +=1;
        m_mutex->unlock();
        t.Stamp();
        Utils::DataImage image = m_dataset->GenerateImage(i);
        cv::Mat result_image(image.Height(), image.Width(), CV_8SC1);
        m_mutex->lock();
        (*m_load_counter) +=t.Elapsed();
        m_mutex->unlock();
        t.Stamp();
        m_forest->getAnnotation(image, &result_image);
        m_mutex->lock();
        (*m_inference_counter) += t.Elapsed();
        m_mutex->unlock();
        t.Stamp();
        // std::cout << "Time for loading: " << t.Elapsed() << std::endl;
        cv::Mat result_bgr = m_dataset->GetColorCoding()->LabelToBgr(result_image);
        cv::imwrite(image.GetResultFilename(), result_bgr);
        m_mutex->lock();
        (*m_total_counter) += t.TotalElapsed();
        ++(*m_progress);
        m_mutex->unlock();
      }
    }else{ //Save the actual unary potentials
      for ( int i = r.begin(); i != r.end(); i++ ) {
        Utils::TimeStamp t;
        m_mutex->lock();
        (*m_image_counter) +=1;
        m_mutex->unlock();
        t.Stamp();
        Utils::DataImage image = m_dataset->GenerateImage(i);
        cv::Mat result_unary;
        m_mutex->lock();
        (*m_load_counter) +=t.Elapsed();
        m_mutex->unlock();
        t.Stamp();
        m_forest->GetUnaryPotential(image, &result_unary, false);
        m_mutex->lock();
        (*m_inference_counter) += t.Elapsed();
        m_mutex->unlock();
        t.Stamp();
        //std::cout << "Time for loading: " << t.Elapsed() << std::endl;
        //result_unary.Save(image.GetUnaryFilename());
        Utils::SaveMat(image.GetUnaryFilename(), result_unary);
        m_mutex->lock();
        (*m_total_counter) += t.TotalElapsed();
        ++(*m_progress);
        m_mutex->unlock();
      }
    }
  }

};

int main(int argc, char *argv[]) {
  //Load configuration data
  Utils::Configuration configuration(argv[1]);
  int dataset_type = configuration.read<int>("dataset_type");
  std::string forest_filename = configuration.read<std::string>("forest_filename");
  bool get_unaries = configuration.read<bool>("get_unaries");
  int class_count = configuration.read<int>("num_classes");

  //Load the forest data, the actual forest for evaluation is loaded later on.
  std::cout << "(test) Loading the Randomized Decision Forest" << std::endl;
  Rdfs::Forest *random_forest = Rdfs::Forest::Load(forest_filename);
  random_forest->PrintForest();
  int requirements = random_forest->GetFeatureRequirements();
  delete random_forest;

  //Load the dataset
  std::cout << "(test) Loading the dataset" << std::endl;
  Utils::StaticDataset * dataset = Utils::StaticDataset::LoadDataset(configuration, Utils::StaticDataset::TEST, requirements);
  Utils::TimeStamp t;
  t.Stamp();
  //Start the evaluation.
  std::cout << "(test) Evaluating images" << std::endl;
  boost::progress_display progress(dataset->ImageCount());
  tbb::parallel_for(tbb::blocked_range<int>(0, dataset->ImageCount(), dataset->ImageCount()/CPUS+1), ParallelEvaluator(dataset, forest_filename, &progress, get_unaries, class_count, dataset_type));
  std::cout << "Time elapsed: " << t.Elapsed() << std::endl;
  delete dataset;
  return EXIT_SUCCESS;
}
