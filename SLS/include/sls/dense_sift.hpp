#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "sls_options.hpp"

struct DescriptorGrid {
    cv::Mat dpMat;
    int numPoints;
    int s1, s2;
};

DescriptorGrid generateDescriptors(const cv::Mat& grayImage, const SLSOptions& opts);
