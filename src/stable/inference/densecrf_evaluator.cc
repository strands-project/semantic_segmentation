/**
 * @file   densecrf_evaluator.cc
 * @author Alexander Hermans
 * @date   Tue Jan 08 2013 11:26pm
 *
 * @brief  CRF evaluation class. Takes one data image and evaluates the dense crf.
 *
 */

#include <cstdio>
#include <string>
#include <cmath>

// Eigen includes
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

// STL includes
#include <fstream>
#include <iostream>
#include <vector>
#include <deque>

// Local includes
#include "densecrf_evaluator.hh"
#include "densecrf.hh"

//Util includes
#include <utils/time_stamp.hh>
#include <utils/configuration.hh>
#include <utils/dataset.hh>

#include <opencv2/highgui/highgui.hpp>

using namespace Inference;

DenseCRFEvaluator::DenseCRFEvaluator(){
}

DenseCRFEvaluator::DenseCRFEvaluator(std::string config_filename){
  Utils::Configuration configuration(config_filename);

  //Set intrinsic params!
  std::string rgb_calib_filename = configuration.read<std::string>("color_calibration_filename");
  std::ifstream in(rgb_calib_filename.c_str());
  if (!in) {
    std::cerr << "Cannot open file " << rgb_calib_filename << std::endl;
  }
  float tmp;
  in >> tmp; m_fx_rgb = tmp;//(0,0)
  in >> tmp; //(0,1);
  in >> tmp; m_cx_rgb = tmp;//(0,2);
  in >> tmp; //(1,0);
  in >> tmp; m_fy_rgb = tmp;//(1,1);
  in >> tmp; m_cy_rgb = tmp;//(1,2);
  m_fx_rgb_inv = 1.0f/m_fx_rgb;
  m_fy_rgb_inv = 1.0f/m_fy_rgb;
  in.close();


  m_smoothing_weight = configuration.read<float>("smoothing_weight");
  m_color_term_weight = configuration.read<float>("color_term_weight");
  m_depth_term_weight = configuration.read<float>("depth_term_weight");
  m_pixel_sigma = configuration.read<float>("pixel_sigma");
  m_longe_range_pixel_sigma = configuration.read<float>("long_range_pixel_sigma");
  m_depth_sigma = configuration.read<float>("depth_sigma");
  m_depth_sigma_long_range = configuration.read<float>("depth_sigma_long_range", 0.1f);
  m_color_sigma = configuration.read<float>("color_sigma");
  m_depth_color_sigma = configuration.read<float>("color_sigma_depth");
  m_normal_sigma = configuration.read<float>("normal_sigma");
  m_iteration_count = configuration.read<unsigned int>("iteration_count");
  m_use_normals = configuration.read<bool>("use_normals");
  m_class_count = configuration.read<int>("num_classes");
  m_use_consistency_step = configuration.read<bool>("consistency_step", false);
  m_use_center = configuration.read<bool>("consistency_center_frame", false);
  //m_consistency_window = configuration.read<int>("reconstruction_consistency_window",1);
  m_consistency_window = 10;
  m_consistency_window_subsampling = configuration.read<int>("reconstruction_subsampling", 1);
  m_do_not_parallelize = false;

}

DenseCRFEvaluator::~DenseCRFEvaluator(){
}

void DenseCRFEvaluator::Evaluate(const Utils::DataImage &image, cv::Mat &segmentation_result) const{
  int W = image.Width();
  int H = image.Height();

  const unsigned char * im = image.GetRGBImage().ptr<unsigned char>(0);
//  const unsigned char * im = image.GetLABImage().Data();
  const float * unary = image.GetUnary().ptr<float>(0);
  //std::cout << image.GetUnary().Width() << "  " << image.GetUnary().Height()<< "  " << image.GetUnary().Bands() << std::endl;
  // Setup the CRF model
  DenseCRF2D crf(W, H, m_class_count);
  if(m_do_not_parallelize){
    crf.StopParallelization();
  }
  crf.setUnaryEnergy( unary );
  crf.addPairwiseGaussian( m_pixel_sigma, m_pixel_sigma, m_smoothing_weight);
  crf.addPairwiseBilateral(m_longe_range_pixel_sigma, m_longe_range_pixel_sigma,m_color_sigma, m_color_sigma, m_color_sigma, im, m_color_term_weight); // <- original simple dense crf... seems to work better than expected!

  // Do map inference
  short * map = new short[W*H];
  crf.map(m_iteration_count, map);
  int index=0;
  for(int y=0; y < H; y++){
    signed char* result_ptr = segmentation_result.ptr<signed char>(y);
    for(int x=0; x < W; ++x, ++result_ptr){
      *result_ptr = map[index];
      index++;
    }
  }
  delete[] map;

}

void DenseCRFEvaluator::Evaluate3D(std::vector<Cluster3D> &cluster_data, int frame_index) const{
  //Setup everything. (Think of consistency window here!
  std::vector<float> x3(cluster_data.size());
  std::vector<float> y3(cluster_data.size());
  std::vector<float> z3(cluster_data.size());
  std::vector<float> n_x3(cluster_data.size());
  std::vector<float> n_y3(cluster_data.size());
  std::vector<float> n_z3(cluster_data.size());
  std::vector<unsigned char> r(cluster_data.size());
  std::vector<unsigned char> g(cluster_data.size());
  std::vector<unsigned char> b(cluster_data.size());
  std::vector<int> index(cluster_data.size());
  int point_counter=0;
  int unary_peak_pointer=0;
  float * unary  = new float[cluster_data.size()*m_class_count];
  for(int i=0; i < cluster_data.size(); i++){
    if((frame_index -cluster_data[i].GetLastFrameIndex()) < m_consistency_window){
      index[point_counter] =i;
      float x,y,z;
      cluster_data[i].GetPosition(&x,&y,&z);
      x3[point_counter] = x;
      y3[point_counter] = y;
      z3[point_counter] = z;

      float n_x,n_y,n_z;
      cluster_data[i].GetNormal(&n_x,&n_y,&n_z);
      n_x3[point_counter] = n_x;
      n_y3[point_counter] = n_y;
      n_z3[point_counter] = n_z;


      unsigned char r_c,g_c,b_c;
      cluster_data[i].GetColor(&r_c,&g_c,&b_c);
      r[point_counter] = r_c;
      g[point_counter] = g_c;
      b[point_counter] = b_c;

      if(!cluster_data[i].IsAccuEmpty()){ // Only update this point if new information was available.
        const std::vector<float> &dist = cluster_data[i].GetAccuDistribution();
        float sum = 1.0f/cluster_data[i].GetAccuDistributionSum();
        for(int c=0; c < m_class_count; c++){
          if(isnan(dist[c])){
            throw std::runtime_error("accu dist is nan!");
          }
          unary[unary_peak_pointer] = -log(dist[c]*sum);
          if(unary[unary_peak_pointer] > 1000){
            unary[unary_peak_pointer] = 1000;
          }

          if(isnan(unary[unary_peak_pointer])){
            throw std::runtime_error("unary is nan!");
          }
          unary_peak_pointer++;
        }
        point_counter++;
      }else{
        const std::vector<float> &dist = cluster_data[i].GetDistribution();
        float sum = 1.0f/2.0f;
        for(int c=0; c < m_class_count; c++){
          if(isnan(dist[c])){
            throw std::runtime_error("accu dist is nan!");
          }
          unary[unary_peak_pointer] = -log((dist[c]+1.0f/static_cast<float>(m_class_count))*sum);
          if(unary[unary_peak_pointer] > 1000){
            unary[unary_peak_pointer] = 1000;
          }

          if(isnan(unary[unary_peak_pointer])){
            throw std::runtime_error("unary is nan!");
          }
          unary_peak_pointer++;
        }
        point_counter++;
      }
    }/*else{
      //Also use these points, however we cannot update them later on, so we will just use what we have now and make it less certain.
    //Then it has some influence on the other points, however not too much and it gets some influence from the other points as well.
    //Thus in theorie resulting in a lot smoother results. Without the "buffer" problem.
    //
    index[point_counter] =i;
      float x,y,z;
      cluster_data[i].GetPosition(&x,&y,&z);
      x3[point_counter] = x;
      y3[point_counter] = y;
      z3[point_counter] = z;

      float n_x,n_y,n_z;
      cluster_data[i].GetNormal(&n_x,&n_y,&n_z);
      n_x3[point_counter] = n_x;
      n_y3[point_counter] = n_y;
      n_z3[point_counter] = n_z;


      unsigned char r_c,g_c,b_c;
      cluster_data[i].GetColor(&r_c,&g_c,&b_c);
      r[point_counter] = r_c;
      g[point_counter] = g_c;
      b[point_counter] = b_c;

      const std::vector<float> &dist = cluster_data[i].GetDistribution();
      float sum = 1.0f/(cluster_data[i].GetDistributionSum()+1.0f);
    float constant = 1.0f/static_cast<float>(m_class_count);
      for(int c=0; c < m_class_count; c++){
        unary[unary_peak_pointer] = -log((dist[c]+constant)*sum);
        if(unary[unary_peak_pointer] > 1000){
          unary[unary_peak_pointer] = 1000;
        }
        unary_peak_pointer++;
      }
      point_counter++;




   }*/

  }
  // std::cout << "Using: " << point_counter << "/" << cluster_data.size() << " points" << std::endl;
  //Do inference.
  float * result  = new float[x3.size()*m_class_count];
  DenseCRF2D crf_3d(1,x3.size(), m_class_count);
  crf_3d.setUnaryEnergy(unary);
  crf_3d.Add3dAppearance(m_depth_sigma_long_range, m_depth_color_sigma, x3.data(), y3.data(), z3.data(), r.data(), g.data(), b.data(), m_depth_term_weight);
  crf_3d.Add3dNormals(m_depth_sigma, m_normal_sigma , x3.data(), y3.data(), z3.data(), n_x3.data(), n_y3.data(), n_z3.data(), m_depth_term_weight);
  //  crf_3d.inference(m_iteration_count,result);
  crf_3d.inference(1,result);

  //Update points that we want to update.
  std::vector<float> temp(m_class_count);
  std::vector<float> inverse_prior(m_class_count, 1.0f/static_cast<float>(m_class_count));

  for(int p=0; p < point_counter; p++){
    for(int c=0; c < m_class_count; c++){
      temp[c] = result[p*m_class_count+c];
      if(isnan(temp[c])){
        std::cout << unary[p*m_class_count+c] << " ";
      }
    }
    if(!cluster_data[index[p]].IsAccuEmpty()){
      const std::vector<float> &current_dist = cluster_data[index[p]].GetAccuDistribution();
      float acc_sum = 1.0f/cluster_data[index[p]].GetAccuDistributionSum();
      std::vector<float> dist(m_class_count,0);

      float sum =0.0f;
      for(int l=0; l < m_class_count; ++l){
        dist[l] = (temp[l]+0.0001f)*inverse_prior[l]*(current_dist[l]*acc_sum+0.0001f) ;
        sum+= dist[l];
      }

      sum = 1.0f/sum;
      for(int l=0; l < m_class_count; ++l){
        dist[l] *= sum;
      }
      cluster_data[index[p]].SetPointDistribution(dist);
    }else{
      const std::vector<float> &current_dist = cluster_data[index[p]].GetDistribution();
      std::vector<float> dist(m_class_count,0);

      float sum =0.0f;
      for(int l=0; l < m_class_count; ++l){
        dist[l] = (temp[l]+0.0001f)*inverse_prior[l]*(current_dist[l]+0.0001f) ;
        sum+= dist[l];
      }

      sum = 1.0f/sum;
      for(int l=0; l < m_class_count; ++l){
        dist[l] *= sum;
      }
      cluster_data[index[p]].SetPointDistribution(dist);
    }

  }

  // clean up
  delete[] result;
  delete[] unary;
}

// void DenseCRFEvaluator::Evaluate2D(const Utils::DataImage &image, Graphics::Image<float> &segmentation_result) const{
//   int W = image.Width();
//   int H = image.Height();
// 
//   const unsigned char * im = image.GetRGBImage().ptr<unsigned char>(0);
//   const float * unary = image.GetUnary().ptr<float>(0);
//   //std::cout << image.GetUnary().Width() << "  " << image.GetUnary().Height()<< "  " << image.GetUnary().Bands() << std::endl;
//   // Setup the CRF model
//   DenseCRF2D crf(W, H, m_class_count);
//   if(m_do_not_parallelize){
//     crf.StopParallelization();
//   }
//   crf.setUnaryEnergy( unary );
//   crf.addPairwiseGaussian( m_pixel_sigma, m_pixel_sigma, m_smoothing_weight);
//   crf.addPairwiseBilateral(m_longe_range_pixel_sigma, m_longe_range_pixel_sigma,m_color_sigma, m_color_sigma, m_color_sigma, im, m_color_term_weight); // <- original simple dense crf... seems to work better than expected!
// 
//   // Do inference
// 
//   crf.inference(m_iteration_count, segmentation_result.Data());
// 
// }

/*void DenseCRFEvaluator::Evaluate(const std::deque<Utils::DataImage> &images, const std::vector<Cluster3D> &cluster_data, int frame_index, Graphics::Image<signed char> &segmentation_result) const{
  int W = images[0].Width();
  int H = images[0].Height();

  int window_size = images.size();

  // Setup the CRF models
  std::vector<DenseCRF2D *> crfs;
  for(int i=0; i< window_size; ++i){
    crfs.push_back(new DenseCRF2D(W,H, m_class_count));
    crfs.back()->setUnaryEnergy(images[i].GetUnary().Data());
    crfs.back()->addPairwiseGaussian( m_pixel_sigma, m_pixel_sigma, m_smoothing_weight);
    crfs.back()->addPairwiseBilateral(m_longe_range_pixel_sigma, m_longe_range_pixel_sigma,m_color_sigma, m_color_sigma, m_color_sigma, images[i].GetRGBImage().Data(), m_color_term_weight);

  }
  // std::cout << "setting up all the crfs is done!" << std::endl;
  float max_penalty = 1.0f;
  int array_size = H*W* m_class_count;

  //setup arrays to handle the penalties.
  std::vector< float * > penalties(crfs.size());
  for(unsigned int d=0; d < crfs.size(); ++d){
    penalties[d] = new float[array_size];
    for( int i=0; i < array_size; ++i){
      penalties[d][i] = max_penalty;
    }
  }

  //std::cout << "penalties are initialized" << std::endl;


  // Do map inference

  for(unsigned int d=0; d < crfs.size(); ++d){
    crfs.at(d)->startInference();
  }

  //std::cout << "started inference!" << std::endl;



  for( int it=0; it<m_iteration_count; it++ ){
    // Do computations here regarding the penalties.

    //collect data
    std::vector< const float *> current(crfs.size());
    for(unsigned int d=0; d < crfs.size(); ++d){
      current[d] = crfs.at(d)->getCurrentProbability();
    }
    // std::cout << "collected data." << std::endl;
    if(window_size > 2){
      //Compute the actual penalties.
      //Go over each cluster
      for(unsigned int cls =0 ; cls < cluster_data.size(); ++cls) {
        //Check the frames we want to include and collect the current labeling.
        std::vector<int> class_distribution(m_class_count,0);
        int total=0;
        for( int f=cluster_data[cls].GetClusterSize(); f>=0; --f ){
          //      std::cout << frame_index << "  " << window_size << "   ->" << cluster_data[cls].image_index[f] << std::endl;
          if( cluster_data[cls].GetImageIndexAtPos(f) <= frame_index - window_size){
            break; // we only consider the frames in the window size.
          }else{//Add the info the class distribution.
            int current_index = cluster_data[cls].GetImageIndexAtPos(f) - frame_index + window_size -1;
            const float* p=&(current[current_index][((cluster_data[cls].GetXCoordAtPos(f)) + W*(cluster_data[cls].GetYCoordAtPos(f)))*m_class_count]);
            int max_index = -1;
            float max =0;
            for(int m=0; m< m_class_count; ++m){
              if(p[m] > max){
                max = p[m];
                max_index = m;
              }
            }
            class_distribution[max_index]++;
            total++;
          }
        }
        if(total >2){
          //   std::cout << "class distribtution is done!" << std::endl;
          //If desired, penalize the data.
          int max_index = -1;
          float max =0;
          for(int m=0; m< m_class_count; ++m){
            if(class_distribution[m] > max){
              max = class_distribution[m];
              max_index = m;
            }
          }
          //    std::cout << "max is " << max_index << std::endl;
          if(static_cast<float>(max)/static_cast<float>(total) > 0.5f ){
            for( int f=cluster_data[cls].GetClusterSize()-1; f>=0; --f ){
              if( cluster_data[cls].GetImageIndexAtPos(f) <= frame_index - window_size){
                break; // we only consider the frames in the window size.
              }else{//Add the info the class distribution.
                int current_index = cluster_data[cls].GetImageIndexAtPos(f) - frame_index + window_size -1;
                float* p=&(penalties[current_index][((cluster_data[cls].GetXCoordAtPos(f)) + W*(cluster_data[cls].GetYCoordAtPos(f)))*m_class_count]);
                p[max_index] *= (1.0f - static_cast<float>(max)/static_cast<float>(total));
              }
            }
          }
        }
        //    std::cout << "penalty is set for this cluster" << std::endl;

      }
      //    std::cout << "Finished setting penalties" << std::endl;
    }

    //Do another step.
    for(unsigned int d=0; d < crfs.size(); ++d){
      crfs.at(d)->stepInference(penalties[d]);
    }

    //  std::cout << "step is done!" << std::endl;
  }


  //Get the final result.
  const float * prob = crfs.back()->getCurrentProbability();

  // Find the map
  int index=0;
  for(int y=0; y < H; y++){
    for(int x=0; x < W; x++){
      const float * p = prob + index*m_class_count;
      // Find the max and subtract it so that the exp doesn't explode
      float mx = p[0];
      int imx = 0;
      for( int j=1; j<m_class_count; j++ )
        if( mx < p[j] ){
          mx = p[j];
          imx = j;
        }
      segmentation_result(x,y)  = imx;
      index++;
    }
  }

  for(unsigned int i=0; i < crfs.size(); ++i){
    delete crfs.at(i);
    delete penalties.at(i);
  }

}

*/

/*void DenseCRFEvaluator::Evaluate3D(const std::deque<Utils::DataImage> &images, std::vector<Cluster3D> &cluster_data, int frame_index, Graphics::Image<signed char> &segmentation_result) const{
  int W = images[0].Width();
  int H = images[0].Height();

  int window_size = (m_consistency_window-1) * m_consistency_window_subsampling +1;


  //===================================================================================================
  //    Collect 3D data
  //===================================================================================================
  //Pass over the clusters once, collecting all relevant data.
  std::vector< std::vector< int> > relevant_cluster_idx(cluster_data.size());
  for(unsigned int i=0; i < cluster_data.size(); ++i){
    //Check this cluster, find the relevant indicis and store them.
    int img=0;
    //setup an index vector.
    std::vector<int> indicis(m_consistency_window+1, -1);
    //traverse all fused points
    for(int j=0; j < cluster_data[ i].GetClusterSize(); ++j){
      img = cluster_data[ i].GetImageIndexAtPos(j) - frame_index + window_size -1;
      if( img < 0){
        //we can stop collecting, only older fusions are coming now.
        break;
      }
      if( img % m_consistency_window_subsampling ==0 ){
        //This is relevant!
        indicis[img/m_consistency_window_subsampling] = j;
        //set last int in this vector to the last frame index to signal this is relevant!
        if(indicis[m_consistency_window]==-1){
          indicis[m_consistency_window]=j;
        }
      }
    }
    relevant_cluster_idx[i] = indicis;
  }
  //===================================================================================================

  //  const float equal = -log(1.0f/m_class_count);
  //  Graphics::Image<float> temp_unary(W,H, m_class_count);
  //  temp_unary.Fill(equal);
  std::vector<Graphics::Image<float> > unary_2d(m_consistency_window);


  for(int w=0; w < m_consistency_window; w++ ){
    //    const float * current =images[w*m_consistency_window_subsampling].GetUnary().Data();
    //    for( int i=0; i < W*H*m_class_count; ++i){
    //      float temp_val = -log(0.2f * unary_2d[w].Data()[i] + 0.8f * exp(-current[i]));
    //      unary_2d[w].Data()[i] = temp_val > 1000 ? 1000 : temp_val;
    //    }
    unary_2d[w] = images[w*m_consistency_window_subsampling].GetUnary();
  }



  //===================================================================================================
  //setup the 3D crf.
  //===================================================================================================
  std::cout << "Points in current cloud: " << cluster_data.size() << std::endl;
  std::vector<float> x3;
  std::vector<float> y3;
  std::vector<float> z3;
  std::vector<float> zeros;
  std::vector<unsigned char> r;
  std::vector<unsigned char> g;
  std::vector<unsigned char> b;
  float * unary  = new float[cluster_data.size()*m_class_count];
  int unary_index=0;
  int x,y, img=0;
  //gather the data from the 3D points, get the RDF unary for each of the pixels.
  for(unsigned int i=0; i < cluster_data.size(); ++i){
    int pos = relevant_cluster_idx[i][m_consistency_window];
    if(pos != -1){
      //this is relevant!
      img = cluster_data[i].GetImageIndexAtPos(pos)  - frame_index + window_size -1;
      x3.push_back(cluster_data[i].GetX());
      y3.push_back(cluster_data[i].GetY());
      z3.push_back(cluster_data[i].GetZ());
      zeros.push_back(0);
      x = cluster_data[i].GetXCoordAtPos(pos);
      y = cluster_data[i].GetYCoordAtPos(pos);

      const Graphics::Image<unsigned char>& rgb = images[img].GetRGBImage();
      r.push_back(rgb(x,y,0));
      g.push_back(rgb(x,y,1));
      b.push_back(rgb(x,y,2));
      std::vector<float> current_unary(m_class_count,0);

      //collect all relevant unaries.
      float count=0;
      for(int w=m_consistency_window-1; w >=0; w--){
        pos = relevant_cluster_idx[i][w];
        if(pos != -1){
          img = cluster_data[i].GetImageIndexAtPos(pos) - frame_index + window_size -1;
          x = cluster_data[i].GetXCoordAtPos(pos);
          y = cluster_data[i].GetYCoordAtPos(pos);

//          float total=0;
//          for(int c=0; c< m_class_count; ++c){
//            float etb = 0.8f*unary_2d[img/m_consistency_window_subsampling](x,y,c);
//            if(etb> 1000){
//              etb=1000;
//            }
//            float rf = 0.2f*-log(cluster_data[i].Get3dUnary(c));
//            if(rf > 1000){
//              rf = 1000;
//            }

//            unary_2d[img/m_consistency_window_subsampling](x,y,c) = rf +etb;
//            current_unary[c] += exp(-unary_2d[img/m_consistency_window_subsampling](x,y,c));
            //float harmonic_mean = exp(-2*etb*rf /(etb+rf));
      //      std::cout << " etb: " << etb <<" , rf: " << rf << ", exp(-mean): " << harmonic_mean << std::endl;
           // current_unary[c] = harmonic_mean;
           // total+= harmonic_mean;
//          }
//          total = 1.0f/total;
       //   std::cout << "Total: " << total <<"Distribution: " ;
//          for(int c=0; c< m_class_count; ++c){
//            current_unary[c] = total* current_unary[c];

//            if(-log(current_unary[c]) > 1000){
//              unary_2d[img/m_consistency_window_subsampling](x,y,c) = 1000;
//            }else{
//              unary_2d[img/m_consistency_window_subsampling](x,y,c) = -log(current_unary[c]);
//            }
//     //       std::cout << current_unary[c];
//          }
     //    std::cout << std::endl;


                    for(int c=0; c< m_class_count; ++c){
                      //Combine with the normal unary.
                      current_unary[c] += 0.8f * exp(-unary_2d[img/m_consistency_window_subsampling](x,y,c)) + 0.2*cluster_data[i].Get3dUnary(c);
                      //Store for further use.
                      unary_2d[img/m_consistency_window_subsampling](x,y,c) = -log(current_unary[c]);

                    }
          count++;
        }
      }
      count = 1.0f/count;
      for(int c=0; c< m_class_count; ++c){
        unary[unary_index] = -log(current_unary[c]*count );
        if(unary[unary_index] > 1000)
          unary[unary_index] = 1000;
        unary_index++;
      }
    }
  }
  std::cout << "Points used in 3D CRF: " <<x3.size()  << std::endl;

  DenseCRF2D crf_3d(1,x3.size(), m_class_count);
  crf_3d.setUnaryEnergy(unary);
  // crf.Add3dSmoothness(m_depth_sigma, x3.data(), y3.data(), z3.data(), m_smoothing_weight);
  crf_3d.Add3dAppearance(m_depth_sigma_long_range, m_depth_color_sigma, x3.data(), y3.data(), z3.data(), r.data(), g.data(), b.data(), m_depth_term_weight);
  //crf.Add3dAppearance(m_depth_sigma_long_range, m_color_sigma, m_normal_sigma, x3.data(), y3.data(), z3.data(), r.data(), g.data(), b.data(), n_x3.data(), n_y3.data(), n_z3.data(), m_color_term_weight);

  crf_3d.startInference();

  //===================================================================================================



  //===================================================================================================
  //setup the 2D crfs.
  //===================================================================================================
  std::vector<DenseCRF2D *> crfs_2d(m_consistency_window);

  for(int w=0; w < m_consistency_window; w++ ){
    crfs_2d[w] = new DenseCRF2D(W, H, m_class_count);
    crfs_2d[w]->setUnaryEnergy( unary_2d[w].Data());
    crfs_2d[w]->addPairwiseGaussian( m_pixel_sigma, m_pixel_sigma, m_smoothing_weight);
    crfs_2d[w]->addPairwiseBilateral(m_longe_range_pixel_sigma, m_longe_range_pixel_sigma,m_color_sigma, m_color_sigma, m_color_sigma, images[w*m_consistency_window_subsampling].GetRGBImage().Data(), m_color_term_weight);
    crfs_2d[w]->startInference();
  }
  //===================================================================================================



  //===================================================================================================
  // Start the main iterations
  //===================================================================================================
  //Set the current 3d Q to the initial value ( = unary)
  float * current_3d_q  = new float[x3.size()*m_class_count];
  const float * temp = crf_3d.getCurrentProbability();
  for(unsigned int i=0; i < x3.size()* m_class_count; ++i){
    current_3d_q[i] = temp[i];
  }


  //Setup a vector to store the current 2d Q values.
  Graphics::Image<float> temp_image(W, H, m_class_count);
  temp_image.Fill(0.0f);
  std::vector<Graphics::Image<float> > additional_3d_messages(m_consistency_window);
  for(int w=0; w< m_consistency_window; w++){
    additional_3d_messages[w] = temp_image;
  }

  for(int k=0; k < m_iteration_count; k++){

    //    if(k==m_iteration_count-10){
    //      //reset unaries
    //      unary_index =0;
    //      for(int i=0; i < cluster_data.size(); ++i){
    //        for(int c=0; c< m_class_count; ++c){
    //          unary[unary_index] = -log(1.0f/m_class_count);
    //          unary_index++;
    //        }
    //      }
    //      crf_3d.setUnaryEnergy(unary);


    ////      for(int w=0; w < m_consistency_window; w++ ){
    ////        temp_image.Fill(-log(1.0f/m_class_count));
    ////        crfs_2d[w]->setUnaryEnergy( temp_image.Data());
    ////      }
    //    }


    //Collect the messages from the 3D dcrf
    const float * messages_3d = crf_3d.computeMessages(current_3d_q);

    //    for(int i=0; i < x3.size(); i+=100){
    //      for(int c=0; c < m_class_count; ++c){
    //        std::cout << messages_3d[i*m_class_count +c] << " ";
    //      }
    //      std::cout << std::endl;
    //    }

    int index=0;
    for(unsigned int i=0; i < cluster_data.size(); ++i){
      int pos = relevant_cluster_idx[i][m_consistency_window];
      if(pos != -1){ //this is relevant!

        //check all the possibly relevant images
        for(int w=m_consistency_window-1; w >=0; w--){
          pos = relevant_cluster_idx[i][w];
          //check if this is relevant
          if(pos!=-1){
            img = (cluster_data[i].GetImageIndexAtPos(pos) - frame_index + window_size -1)/m_consistency_window_subsampling;
            //set the unary
            x = cluster_data[i].GetXCoordAtPos(pos);
            y = cluster_data[i].GetYCoordAtPos(pos);
            for(int c=0; c< m_class_count; ++c){
              additional_3d_messages[img](x,y,c) =  messages_3d[index*m_class_count + c];
              //  std::cout << img << " " << x << " " << y << " " << messages_3d[index*m_class_count + c] << std::endl;
            }
          }
        }
        index++;
      }
    }

    //    for(int w=m_consistency_window-1; w >=0; w--){
    //      for(int x=0; x < W; ++x){
    //        for(int y=0; y < H; ++y){
    //          for(int c=0; c < m_class_count; ++c){
    //            std::cout << additional_3d_messages[w](x,y,c) << " ";
    //          }
    //          std::cout << std::endl;
    //        }
    //      }
    //    }


    //Run dense CRFs in 2D
    for(int w=0; w < m_consistency_window; w++ ){
      crfs_2d[w]->stepInference(additional_3d_messages[w].Data());
    }


    //Collect current Q from 2D
    int q_index=0;
    for(unsigned int i=0; i < cluster_data.size(); ++i){
      int pos = relevant_cluster_idx[i][m_consistency_window];
      if(pos != -1){
        //this is relevant!
        std::vector<float> new_q_3d(m_class_count,0);
        float count=0;
        for(int w=m_consistency_window-1; w >=0; w--){
          pos = relevant_cluster_idx[i][w];
          if(pos!=-1){
            x = cluster_data[i].GetXCoordAtPos(pos);
            y = cluster_data[i].GetYCoordAtPos(pos);
            img = (cluster_data[i].GetImageIndexAtPos(pos) - frame_index + window_size -1)/m_consistency_window_subsampling;
            const float * current_q_2d_crf = crfs_2d[img]->getCurrentProbability();
            for(int c=0; c< m_class_count; ++c){
              new_q_3d[c] += current_q_2d_crf[(x+W*y)*m_class_count+c];
            }
            count++;
          }
        }
        count = 1.0f/count;
        for(int c=0; c< m_class_count; ++c){
          current_3d_q[q_index] = new_q_3d[c]*count;
          q_index++;
        }
      }
    }

    //        for(int i=0; i < x3.size(); i+=100){
    //          for(int c=0; c < m_class_count; ++c){
    //            std::cout << current_3d_q[i*m_class_count +c] << " ";
    //          }
    //          std::cout << std::endl;
    //        }

  }



  int image_pos = m_consistency_window-1;

  //Setup variables to work on the center image.
  if(m_use_center){
    image_pos =m_consistency_window*0.5f;
    //      image_pos = start;
  }
  const float* result = crfs_2d[image_pos]->getCurrentProbability();
  int index2=0;
  for(int y=0; y < H; y++){
    for(int x=0; x < W; x++){
      int max_label= -1;
      float max_prob=0;
      for(int c=0; c < m_class_count; ++c){
        if(result[index2] > max_prob){
          max_prob = result[index2];
          max_label = c;
        }
        index2++;
      }
      segmentation_result(x,y) = max_label;
    }
  }

  //Assign labels to the points.
  unary_index=0;
  for(unsigned int i=0; i < cluster_data.size(); ++i){
    int pos = relevant_cluster_idx[i][m_consistency_window];
    if(pos != -1){
      //this is relevant!
      std::vector<float> current_unary(m_class_count,0);
      float count=0;
      for(int w=m_consistency_window-1; w >=0; w--){
        pos = relevant_cluster_idx[i][w];
        if(pos!=-1){
          x = cluster_data[i].GetXCoordAtPos(pos);
          y = cluster_data[i].GetYCoordAtPos(pos);
          img = (cluster_data[i].GetImageIndexAtPos(pos) - frame_index + window_size -1)/m_consistency_window_subsampling;
          const float * unary_crf = crfs_2d[img]->getCurrentProbability();
          for(int c=0; c< m_class_count; ++c){
            current_unary[c] += unary_crf[(x+W*y)*m_class_count+c];
          }
          count++;
        }
      }
      count = 1.0f/count;
      for(int c=0; c< m_class_count; ++c){
        current_unary[c] = current_unary[c]*count;
      }
      cluster_data[i].AddDistribution(current_unary);
      cluster_data[i].UpdateLabel();
    }
  }



  // Clean up
  for(int i=0; i < m_consistency_window; i++){
    delete crfs_2d.at(i);
  }

  delete[] unary;
  delete[] current_3d_q;






}*/

void DenseCRFEvaluator::StopParallelization(){
  m_do_not_parallelize = true;
}

int DenseCRFEvaluator::GetLoadRequirements(){
  return Utils::RGB | Utils::UNARY;
  //return Utils::LAB | Utils::UNARY;
}
