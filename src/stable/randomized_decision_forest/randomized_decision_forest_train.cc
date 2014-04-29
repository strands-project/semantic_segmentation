// localincludes
#include "forest.hh"
#include <sstream>

// Utils includes
#include <utils/data_image.hh>
#include <utils/configuration.hh>
#include <utils/static_dataset.hh>
#include <utils/random_source.hh>

// Boost includes
#include <boost/progress.hpp>

// TBB includes
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#define CPUS  8

class ParallelLoader {
  boost::progress_display* m_progress;
  std::vector<Utils::DataImage> &m_data_images;
  Utils::StaticDataset * m_dataset;

public:
  ParallelLoader(Utils::StaticDataset * dataset, boost::progress_display* progress,
                 std::vector<Utils::DataImage> &data_images):
                 m_progress(progress), m_data_images(data_images), m_dataset(dataset)
  { }

  void  operator() (const tbb::blocked_range<int>& r) const{
    for ( int i = r.begin(); i != r.end(); i++ ) { // iterates over the entire chunk
      m_data_images[i] = m_dataset->GenerateImage(i);
      ++(*m_progress);
    }
  }

};

int main(int argc, char *argv[]) {
  //Load the configuration
  Rdfs::Forest random_forest(argv[1]);

  //Load the training set
   int requirements = random_forest.GetFeatureRequirements();
  std::cout << "(train) Loading the all filenames" << std::endl;
  Utils::StaticDataset * dataset = Utils::StaticDataset::LoadDataset(argv[1], Utils::StaticDataset::TRAIN, Utils::ANNOTATION | Utils::RGB | requirements);

  // Load the actual images
  std::cout << "(train) Loading the dataset" << std::endl;
  std::vector<Utils::DataImage> data_images(dataset->ImageCount());
  boost::progress_display progress(dataset->ImageCount());
  tbb::parallel_for(tbb::blocked_range<int>(0, dataset->ImageCount(), dataset->ImageCount()/CPUS+1), ParallelLoader(dataset, &progress, data_images));

  //Start the actual training of the forest with this data.
  std::cout << "(train) Training the forest" << std::endl;

  random_forest.Train(data_images);

  random_forest.Save(argv[1]);
}

