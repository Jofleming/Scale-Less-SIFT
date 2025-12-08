#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "sls_options.hpp"

struct DescriptorGrid {
    cv::Mat dpMat;   // D x (numPoints * numSigma), CV_32F
    int numPoints;   // number of grid locations N
    int s1, s2;      // grid dimensions (width, height)
};

DescriptorGrid generateDescriptors(const cv::Mat& grayImage, const SLSOptions& opts);
