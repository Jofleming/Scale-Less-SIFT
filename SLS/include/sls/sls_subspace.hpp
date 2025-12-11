#pragma once
#include <opencv2/core.hpp>

cv::Mat constructBasis(const cv::Mat& X, int subsDim);

cv::Mat computeSLSDescriptors(const cv::Mat& dpMat,
    int numPoints,
    int s1, int s2,
    int numSigma,
    int subsDim);
