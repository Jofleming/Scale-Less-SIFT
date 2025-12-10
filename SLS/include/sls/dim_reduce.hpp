#pragma once
#include <opencv2/core.hpp>
#include "sls_options.hpp"

struct DimReduceResult {
    cv::Mat dpMat1Reduced;
    cv::Mat dpMat2Reduced;
    cv::Mat pcaBasis;
};

DimReduceResult dimReduce(const cv::Mat& dpMat1,
    const cv::Mat& dpMat2,
    const SLSOptions& opts);
