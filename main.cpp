/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: main.cpp
    Last modifed:   29.12.2016 by Leonardo Citraro
    Description:    Basic test of the HOG feature

    =========================================================================
    https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    =========================================================================
*/
#include "HOG.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>
#include <math.h>
#include <chrono>
#include <boost/filesystem.hpp>
#include "csv.hpp"

namespace fs = boost::filesystem;
using namespace std;

int verbose = 1;

int main(int argc, char* argv[])
{
    // IO variables
    fs::path input_path (argv[1]);
    fs::path output_file (argv[2]);

    // Retrieve the HOG from the image
    size_t crop_height = 256; //atoi(argv[3]);
    size_t crop_width = 128; //atoi(argv[4]);
    size_t blocksize = 32; //atoi(argv[5]);
    size_t cellsize = 16; //atoi(argv[6]);
    size_t stride = 16; //atoi(argv[7]);
    size_t binning = 9; //atoi(argv[8]);
    
    // Auxiliary variables for memory allocation of HOG features
    size_t cells_x_grid = blocksize / cellsize;
    size_t hog_bins = binning * (cells_x_grid * cells_x_grid);
    size_t w = (crop_width - blocksize) / stride + 1;
    size_t h = (crop_height - blocksize) / stride + 1;
    size_t hog_size = hog_bins * (w * h);
    
    // Instantiate HOG feature extractor
    HOG hog(blocksize, cellsize, stride, binning, HOG::GRADIENT_UNSIGNED);

    // List image files to be described
    std::vector<std::string> filenames;
    for (fs::directory_iterator itr(input_path); itr != fs::directory_iterator(); ++itr)
    {
        if (fs::is_regular_file(itr->status()))
            filenames.push_back(itr->path().filename().string());
    }
    
    // Init HOG features
    cv::Mat hog_features (filenames.size(), hog_size, CV_32FC1);
    
    // Loop over images (measure time)
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int n = filenames.size();
    for (int i = 0; i < n; i++)
    {
        string filepath = (input_path / fs::path(filenames[i])).string();
        cv::Mat image = cv::imread(filepath, CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat image_crop;
        cv::resize( image, image_crop, cv::Size(crop_width, crop_height) );
    
        if (verbose > 0)
            cout << '(' << i << '/' << n-1 << ')' << ' ' << filenames[i] << ' ';

        hog.process(image);
        auto hist = hog.retrieve(cv::Rect(0,0,image_crop.cols, image_crop.rows));
    
        assert(hist.size() == hog_size);
        
        for (int k = 0; k < hist.size(); k++)
            hog_features.at<float>(i,k) = hist[k];
        
        if (verbose > 0)
            cout << " -> DONE" << '\n';
    }
    
    cv::FileStorage fs;
    fs.open(output_file.string(), cv::FileStorage::APPEND);
    fs << "filenames"<< filenames;
    fs << "hog_features" << hog_features;
    fs.release();
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    std::cout << "Total elapsed time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<std::endl;

    return 0;
}
