#pragma once


#ifndef RF_HOG_DESCRIPTOR_H
#define RF_HOG_DESCRIPTOR_H


#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;

#include "img_functions.h"


class HOG_Descriptor {

public:

    HOG_Descriptor() {
        //initialize default parameters(win_size, block_size, block_step,....)
        win_size = cv::Size(64, 64);

        //Fill other parameters here
        block_size = cv::Size(16, 16);
        block_step = cv::Size(8, 8);
        cell_size = cv::Size(8, 8);
        nbins = 9;

        // parameter to check if descriptor is already initialized or not
        is_init = false;
    };


    void setWinSize(cv::Size sz) {
        win_size = sz;
    }

    cv::Size getWinSize() {
        return win_size;
    }

    void setBlockSize(cv::Size sz) {
        block_size = sz;
    }

    void setBlockStep(cv::Size sz) {
        block_step = sz;
    }

    void setCellSize(cv::Size sz) {
        cell_size = sz;
    }

    void setPadSize(cv::Size sz) {
        pad_size = sz;
    }


    void initDetector();

    void visualizeHOG(cv::Mat img, std::vector<float>& feats, cv::HOGDescriptor& hog_detector, int scale_factor = 3);

    void detectHOGDescriptor(cv::Mat& im, std::vector<float>& feat, cv::Size sz, bool show);

    ~HOG_Descriptor() {};


private:
    cv::Size win_size; 
    cv::Size block_size;
    cv::Size block_step;
    cv::Size cell_size;
    cv::Size pad_size;
    int nbins;

    cv::HOGDescriptor hog_detector;
public:
    cv::HOGDescriptor& getHog_detector();

private:
    bool is_init;
};

#endif //RF_HOG_DESCRIPTOR_H
