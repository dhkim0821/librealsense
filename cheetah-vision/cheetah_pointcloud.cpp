// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#include <algorithm>            // std::min, std::max


#include <iostream>
#include <thread> 
#include <type_traits>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <librealsense2-gl/rs_processing_gl.hpp> // Include GPU-Processing API
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "cheetah_pointcloud.h"
#include "opencv2/imgcodecs.hpp"
#include <unistd.h>

#include <chrono>

int main(int argc, char * argv[]) try
{
  printf("start poincloud processing\n");
  rs2::pointcloud pc;
  rs2::points points;
  
  rs2::pipeline pipe;
  rs2::config D435cfg;
  D435cfg.enable_stream(RS2_STREAM_DEPTH, 480,270, RS2_FORMAT_Z16, 90);
  //D435cfg.enable_stream(RS2_STREAM_DEPTH, 640,480, RS2_FORMAT_Z16, 90);
  //D435cfg.enable_stream(RS2_STREAM_DEPTH);
  pipe.start(D435cfg);

  LocalizationHandle localizationObject;
  vision_lcm.subscribe("global_to_robot", &LocalizationHandle::handlePose, &localizationObject);
  std::thread localization_thread(&handleLCM);

  // World heightmap initialization
  for(int i(0); i<1000;++i){
    for(int j(0); j<1000; ++j){
      world_heightmap.map[i][j] = 0.;
    }
  }
  // Traversability initialization
  for(int i(0); i<100;++i){
    for(int j(0); j<100; ++j){
      traversability.map[i][j] = 0;
    }
  }

  int iter(0);

  rs2::gl::uploader   upload;     // used to explicitly copy frame to the GPU

  while (true) { 
    ++iter;

    auto start = std::chrono::system_clock::now();

    rs2::frameset D435frames = pipe.wait_for_frames();
    auto depth = D435frames.get_depth_frame();
    depth = upload.process(depth);

    points = pc.calculate(depth);
    _ProcessPointCloudData(points);

    auto end = std::chrono::system_clock::now();

    //if(iter%3000 == 1) printf("point cloud loop is run\n");
    if(iter%10 == 1){
        printf("point cloud loop is run\n");
        double elapsed_time = 
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        printf("time taken: %f (ms)\n", elapsed_time);
    }
  }
  return EXIT_SUCCESS;
}

catch (const rs2::error & e)
{
  std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
  return EXIT_FAILURE;
}
catch (const std::exception & e)
{
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}

void _ProcessPointCloudData(const rs2::points & points){

  static rs_pointcloud_t cf_pointcloud; // 921600
  static rs_pointcloud_t wf_pointcloud; 

  int num_points = points.size();
  auto vertices = points.get_vertices(); 

  // Move raw image into camera frame point cloud struct
  int k(0);
  int num_valid_points(0);
  std::vector<int> valid_indices; 

  for (int i = 0; i< num_points; i++)
  {
    if (vertices[i].z and vertices[i].z < 1.0)
    {
      num_valid_points++;
      valid_indices.push_back(i);
    }
  }

  int num_skip = floor(num_valid_points/1000);
  int maximum_valid_point = 5000;

  if(num_skip == 0){ num_skip = 1; }
  for (int i = 0; i < floor(num_valid_points/num_skip); i++)
  {
    cf_pointcloud.pointlist[k][0] = vertices[valid_indices[i*num_skip]].z;
    cf_pointcloud.pointlist[k][1] = -vertices[valid_indices[i*num_skip]].x;
    cf_pointcloud.pointlist[k][2] = -vertices[valid_indices[i*num_skip]].y;
    ++k;
    if(k>maximum_valid_point){
      break;
    }
  }
  SE3::SE3Multi(global_to_robot, robot_to_D435, global_to_D435);
  global_to_D435.pointcloudTransformation(cf_pointcloud, wf_pointcloud);

  wfPCtoHeightmap(&wf_pointcloud, &world_heightmap, maximum_valid_point); //right
  extractLocalFromWorldHeightmap(global_to_robot.xyz, &world_heightmap, &local_heightmap); // writes over local heightmap in place

  //int cv_type = CV_32F;
  int cv_type = CV_64F;
  cv::Mat cv_local_heightmap(100, 100, cv_type, local_heightmap.map);

  // filter
  int erosion_size = 4;
  static cv::Mat erosion_element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ), 
      cv::Point( erosion_size, erosion_size ) );
  int dilation_size = 4;
  static cv::Mat dilation_element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), 
      cv::Point( dilation_size, dilation_size ) );

  //cv::erode( cv_local_heightmap, cv_local_heightmap, erosion_element );
  cv::erode( cv_local_heightmap, cv_local_heightmap, erosion_element );
  cv::dilate( cv_local_heightmap, cv_local_heightmap, dilation_element );
  //cv_local_heightmap = max(cv_local_heightmap, 0.);
  
  //threshold(gpu_heightmap, gpu_heightmap, 0.0, 255., THRESH_TOZERO);
  //threshold(gpu_heightmap, gpu_heightmap, 0.0, 255., 3);

  cv::Mat	grad_x, grad_y;
  cv::Sobel(cv_local_heightmap, grad_x, cv_type, 1,0,3,1,0,cv::BORDER_DEFAULT);
  cv::Sobel(cv_local_heightmap, grad_y, cv_type, 0,1,3,1,0,cv::BORDER_DEFAULT);
  cv::Mat abs_grad_x = abs(grad_x);
  cv::Mat abs_grad_y = abs(grad_y);
  cv::Mat grad_max = max(abs_grad_x, abs_grad_y);
  cv::Mat no_step_mat, jump_mat;
  cv::threshold(grad_max, no_step_mat, 0.07, 1, 0);
  cv::threshold(grad_max, jump_mat, 0.5, 1, 0);
  cv::Mat traversability_mat(100, 100, CV_32S);
  traversability_mat = no_step_mat + jump_mat;

  //cv::Mat mat;
  //gpu_heightmap.download(mat);

  for(int i(0); i<100; ++i){
      for(int j(0); j<100; ++j){
          local_heightmap.map[i][j] = cv_local_heightmap.at<double>(i,j);
          traversability.map[i][j] = traversability_mat.at<double>(i,j);
      }
  }

  (local_heightmap).robot_loc[0] = global_to_robot.xyz[0];
  (local_heightmap).robot_loc[1] = global_to_robot.xyz[1];
  (local_heightmap).robot_loc[2] = global_to_robot.xyz[2];

  vision_lcm.publish("local_heightmap", &local_heightmap);
  vision_lcm.publish("traversability", &traversability);
  vision_lcm.publish("cf_pointcloud", &wf_pointcloud);
}
