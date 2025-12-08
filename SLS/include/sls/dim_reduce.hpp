#pragma once
#include <opencv2/core.hpp>
#include "sls_options.hpp"

struct DimReduceResult {
    cv::Mat dpMat1Reduced;  // reducedDim x (N1*S)
    cv::Mat dpMat2Reduced;  // reducedDim x (N2*S)
    cv::Mat pcaBasis;       // originalDim x reducedDim
};

DimReduceResult dimReduce(const cv::Mat& dpMat1,
    const cv::Mat& dpMat2,
    const SLSOptions& opts);
