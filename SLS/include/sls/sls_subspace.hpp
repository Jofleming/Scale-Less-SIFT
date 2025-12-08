#pragma once
#include <opencv2/core.hpp>

cv::Mat constructBasis(const cv::Mat& X, int subsDim);

// dpMat: D x (numPoints * numSigma)
// returns SLS descriptor volume: size (s2 x s1) with numElements channels
cv::Mat computeSLSDescriptors(const cv::Mat& dpMat,
    int numPoints,
    int s1, int s2,
    int numSigma,
    int subsDim);
