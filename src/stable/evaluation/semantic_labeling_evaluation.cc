#include <iostream>
#include <iomanip>
#include <vector>

#include <utils/configuration.hh>
#include <utils/static_dataset.hh>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


// Boost includes
#include <boost/progress.hpp>

int main(int argc, char *argv[]) {
  Utils::Configuration configuration(argv[1]);
  int num_classes = configuration.read<int>("num_classes");
  int dataset_type = configuration.read<int>("dataset_type");

  std::cout << "(test) Loading dataset" << std::endl;
  Utils::StaticDataset  * dataset = Utils::StaticDataset::LoadDataset(configuration,Utils::StaticDataset::TEST,   Utils::ANNOTATION | Utils::DEPTH);

  std::cout << "(test) Evaluating results" << std::endl;
  boost::progress_display progress(dataset->ImageCount());

  if(argc >= 3 && std::string(argv[2]).compare(std::string("-cloud")) == 0) {
    //Special case where other negative labels than -1 are allowed.
    num_classes +=2;
    std::vector<int> pixels_total_class(num_classes, 0);
    std::vector<int> pixels_ok_class(num_classes, 0);
    std::vector<int> pixels_label(num_classes, 0);
    std::vector<int> confusion(num_classes * num_classes, 0);

    int pixels_total = 0;
    int pixels_ok = 0;


    for (int i=0; i < static_cast<int>(dataset->ImageCount()); i++) {
      Utils::DataImage current = dataset->GenerateImage(i);
      cv::Mat result = cv::imread(current.GetResultFilename(), CV_LOAD_IMAGE_COLOR);
      cv::Mat result_labels = dataset->GetColorCoding()->BgrToLabel(result);
      cv::Mat annotation = current.GetAnnotation();
      for (int l = 0; l < annotation.rows; l++) {
        for (int k = 0; k < annotation.cols; k++) {
          signed char annotation_label = annotation.at<signed char>(l,k);
          signed char result_label = result_labels.at<signed char>(l,k);
          if(result_label < -1){ // append the -3 and -2 labels the two last classes.
            result_label = num_classes + result_label +1;
          }
          if (annotation_label != -1) {
            if(result_label == num_classes-1){
              //We want to show this, but not  count them as no depth is not evaluated currently.
              pixels_label[result_label]++;
            }else{
              //count as normal label.
              pixels_total++;
              pixels_total_class[annotation_label]++;
              pixels_label[result_label]++;
              if (annotation_label == result_label) {
                pixels_ok++;
                pixels_ok_class[annotation_label]++;
              }
            }
            confusion[annotation_label * num_classes + result_label]++;
          }
        }
      }
      ++progress;
    }
    delete dataset;

    double average = static_cast<double>(0.0);
    double waverage = static_cast<double>(0.0);

    for (int32_t i = 0; i < static_cast<int32_t>(num_classes); i++) {
      average += (pixels_total_class[i]==0) ? 0 : pixels_ok_class[i] / static_cast<double>(pixels_total_class[i]);
      waverage += (pixels_total_class[i] + pixels_label[i] - pixels_ok_class[i] == 0) ? 0 : pixels_ok_class[i] /
                                                                                        static_cast<double>(pixels_total_class[i] + pixels_label[i] - pixels_ok_class[i]);
    }

    average /= static_cast<double>(num_classes-2);
    waverage /= static_cast<double>(num_classes-2);


    std::cout << "|";
    for(int i=0; i < num_classes -2; ++i) std::cout << std::string(8, '-') << "+";
    std::cout <<  std::string(15, '-') << "+"<< std::string(15, '-') << "|"<< std::endl;

    std::cout << "| classes ";
    for(int i=0; i < num_classes -4; ++i) std::cout << std::string(9, ' ');
    std::cout <<  std::string(8, ' ') << "|";
    std::cout << "      no label |      no depth |" << std::endl;


    std::cout << "|";
    for(int i=0; i < num_classes -2; ++i) std::cout << std::string(8, '-') << "+";
    std::cout <<  std::string(15, '-') << "+"<< std::string(15, '-') << "|"<< std::endl;

    for (int32_t q = 0; q < static_cast<double>(num_classes)-2; q++) {
      for (int32_t w = 0; w < static_cast<double>(num_classes); w++) {
        double result = (pixels_total_class[q] == 0) ? 0 : 100.0 * (confusion[q * num_classes + w] / static_cast<double>(pixels_total_class[q]));
        if(w < static_cast<double>(num_classes) -2){
          std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6)<< result << " ";
        }else{
          std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(13)<< result << " ";
        }
      }
      std::cout << "|" << std::endl;
      std::cout << "|";
      for(int i=0; i < num_classes -2; ++i) std::cout << std::string(8, '-') << "+";
      std::cout <<  std::string(15, '-') << "+"<< std::string(15, '-') << "|"<< std::endl;

    }

    std::cout << std::endl;

    std::cout << "|-----------------+";
    for(int i=0; i < num_classes -3; ++i) std::cout << std::string(8, '-') << "+";
    std::cout <<  std::string(8, '-') << "|"<< std::endl;

    std::cout << "| Int. over Union ";



    for (int32_t q = 0; q < static_cast<double>(num_classes-2); q++) {
      double result = (pixels_total_class[q] + pixels_label[q] - pixels_ok_class[q] == 0) ? 0 : 100.0 * pixels_ok_class[q] / static_cast<double>(pixels_total_class[q] + pixels_label[q] - pixels_ok_class[q]);
      std::cout << std::setfill('0');
      std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6)<< result << " ";
    }
    std::cout << "|" << std::endl;
    std::cout << "|-----------------+";
    for(int i=0; i < num_classes -3; ++i) std::cout << std::string(8, '-') << "+";
    std::cout <<  std::string(8, '-') << "|"<< std::endl;

    std::cout << "| Class precision ";
    for (int32_t q = 0; q < static_cast<double>(num_classes-2); q++) {
      double result = (pixels_total_class[q] == 0) ? 0 : 100.0 * pixels_ok_class[q] / static_cast<double>(pixels_total_class[q]);
      std::cout << std::setfill('0');
      std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6)<< result << " ";
    }
    std::cout << "|" << std::endl;
    std::cout << "|-----------------+";
    for(int i=0; i < num_classes -3; ++i) std::cout << std::string(8, '-') << "+";
    std::cout <<  std::string(8, '-') << "|"<< std::endl;

    std::cout << std::endl;
    double result = (pixels_total != 0) ? 100.0 * static_cast<double>(pixels_ok) / static_cast<double>(pixels_total) : 100.0 * static_cast<double>(0.0);
    std::cout << "|----------+--------|" <<std::endl;
    std::cout << "| Overall  | " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6) << result << " |" << std::endl;
    std::cout << "|----------+--------|" <<std::endl;
    std::cout << "| Average  | " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6) << 100.0 * average << " |" << std::endl;
    std::cout << "|----------+--------|" <<std::endl;
    std::cout << "| WAverage | " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6) << 100.0 * waverage << " |" << std::endl;
    std::cout << "|----------+--------|" <<std::endl;

    return EXIT_SUCCESS;

  }else{

    std::vector<int> pixels_total_class(num_classes, 0);
    std::vector<int> pixels_ok_class(num_classes, 0);
    std::vector<int> pixels_label(num_classes, 0);
    std::vector<int> confusion(num_classes * num_classes, 0);

    int pixels_total = 0;
    int pixels_ok = 0;

    for (int i=0; i < static_cast<int>(dataset->ImageCount()); i++) {
      Utils::DataImage current = dataset->GenerateImage(i);
      cv::Mat result = cv::imread(current.GetResultFilename(), CV_LOAD_IMAGE_COLOR);
      cv::Mat result_labels = dataset->GetColorCoding()->BgrToLabel(result);
      cv::Mat annotation= current.GetAnnotation();
      for (int l = 0; l < annotation.rows; l++) {
        for (int k = 0; k < annotation.cols; k++) {
          signed char annotation_label = annotation.at<signed char>(l,k);
          signed char result_label = result_labels.at<signed char>(l,k);
          if(result_label < -1){ // append the -3 and -2 labels the two last classes.
            result_label = num_classes + result_label +1;
          }
          if (annotation_label != -1) {
            pixels_total++;
            pixels_total_class[annotation_label]++;
            pixels_label[result_label]++;
            if (annotation_label == result_label) {
              pixels_ok++;
              pixels_ok_class[annotation_label]++;
            }
            confusion[annotation_label * num_classes + result_label]++;
          }
        }
      }
      ++progress;
    }
    delete dataset;

    double average = static_cast<double>(0.0);
    double waverage = static_cast<double>(0.0);

    for (int32_t i = 0; i < static_cast<int32_t>(num_classes); i++) {
      average += (pixels_total_class[i]==0) ? 0 : pixels_ok_class[i] / static_cast<double>(pixels_total_class[i]);
      waverage += (pixels_total_class[i] + pixels_label[i] - pixels_ok_class[i] == 0) ? 0 : pixels_ok_class[i] /
                                                                                        static_cast<double>(pixels_total_class[i] + pixels_label[i] - pixels_ok_class[i]);
    }

    average /= static_cast<double>(num_classes);
    waverage /= static_cast<double>(num_classes);

    if(argc >= 3 && std::string(argv[2]).compare(std::string("-org")) == 0) {

      std::cout << std::endl<< "Confusion matrix" << std::endl;
      std::cout << "|";
      for(int i=0; i < num_classes -1; ++i) std::cout << std::string(8, '-') << "+";
      std::cout <<  std::string(8, '-') << "|"<< std::endl;

      for (int32_t q = 0; q < static_cast<double>(num_classes); q++) {
        for (int32_t w = 0; w < static_cast<double>(num_classes); w++) {
          double result = (pixels_total_class[q] == 0) ? 0 : 100.0 * (confusion[q * num_classes + w] / static_cast<double>(pixels_total_class[q]));
          std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6)<< result << " ";
        }
        std::cout << "|" << std::endl;
        std::cout << "|";
        for(int i=0; i < num_classes -1; ++i) std::cout << std::string(8, '-') << "+";
        std::cout <<  std::string(8, '-') << "|"<< std::endl;
      }

      std::cout << std::endl;

      std::cout << "|-----------------+";
      for(int i=0; i < num_classes -1; ++i) std::cout << std::string(8, '-') << "+";
      std::cout <<  std::string(8, '-') << "|"<< std::endl;

      std::cout << "| Int. over Union ";



      for (int32_t q = 0; q < static_cast<double>(num_classes); q++) {
        double result = (pixels_total_class[q] + pixels_label[q] - pixels_ok_class[q] == 0) ? 0 : 100.0 * pixels_ok_class[q] / static_cast<double>(pixels_total_class[q] + pixels_label[q] - pixels_ok_class[q]);
        std::cout << std::setfill('0');
        std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6)<< result << " ";
      }
      std::cout << "|" << std::endl;
      std::cout << "|-----------------+";
      for(int i=0; i < num_classes -1; ++i) std::cout << std::string(8, '-') << "+";
      std::cout <<  std::string(8, '-') << "|"<< std::endl;

      std::cout << "| Class precision ";
      for (int32_t q = 0; q < static_cast<double>(num_classes); q++) {
        double result = (pixels_total_class[q] == 0) ? 0 : 100.0 * pixels_ok_class[q] / static_cast<double>(pixels_total_class[q]);
        std::cout << std::setfill('0');
        std::cout << "| " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6)<< result << " ";
      }
      std::cout << "|" << std::endl;
      std::cout << "|-----------------+";
      for(int i=0; i < num_classes -1; ++i) std::cout << std::string(8, '-') << "+";
      std::cout <<  std::string(8, '-') << "|"<< std::endl;

      std::cout << std::endl;
      double result = (pixels_total != 0) ? 100.0 * static_cast<double>(pixels_ok) / static_cast<double>(pixels_total) : 100.0 * static_cast<double>(0.0);
      std::cout << "|----------+--------|" <<std::endl;
      std::cout << "| Overall  | " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6) << result << " |" << std::endl;
      std::cout << "|----------+--------|" <<std::endl;
      std::cout << "| Average  | " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6) << 100.0 * average << " |" << std::endl;
      std::cout << "|----------+--------|" <<std::endl;
      std::cout << "| WAverage | " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setfill(' ')<< std::setw(6) << 100.0 * waverage << " |" << std::endl;
      std::cout << "|----------+--------|" <<std::endl;

    }else{


      for (int32_t q = 0; q < static_cast<double>(num_classes); q++) {
        for (int32_t w = 0; w < static_cast<double>(num_classes); w++) {
          double result = (pixels_total_class[q] == 0) ? 0 : 100.0 * (confusion[q * num_classes + w] / static_cast<double>(pixels_total_class[q]));
          std::cout << std::setfill('0');
          std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setw(2) << result << " ";
        }
        std::cout << std::endl;
      }

      std::cout << std::endl;
      for (int32_t q = 0; q < static_cast<double>(num_classes); q++) {
        double result = (pixels_total_class[q] + pixels_label[q] - pixels_ok_class[q] == 0) ? 0 : 100.0 * pixels_ok_class[q] / static_cast<double>(pixels_total_class[q] + pixels_label[q] - pixels_ok_class[q]);
        std::cout << std::setfill('0');
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setw(2) << result << " ";
      }
      std::cout << std::endl;
      for (int32_t q = 0; q < static_cast<double>(num_classes); q++) {
        double result = (pixels_total_class[q] == 0) ? 0 : 100.0 * pixels_ok_class[q] / static_cast<double>(pixels_total_class[q]);
        std::cout << std::setfill('0');
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setw(2) << result << " ";
      }
      std::cout << std::endl;
      double result = (pixels_total != 0) ? 100.0 * static_cast<double>(pixels_ok) / static_cast<double>(pixels_total) : 100.0 * static_cast<double>(0.0);
      std::cout << std::setfill('0');
      std::cout << "Overall: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setw(2) << result;
      std::cout << ", Average: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setw(2) << 100.0 * average;
      std::cout << ", WAverage: " << std::setiosflags(std::ios::fixed) << std::setprecision(2) << std::setw(2) << 100.0 * waverage << std::endl;
    }
    return EXIT_SUCCESS;
  }
}
