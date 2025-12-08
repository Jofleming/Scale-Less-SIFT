#pragma once
#include <opencv2/core.hpp>
#include "sls_options.hpp"

struct SLSOutput {
    cv::Mat desc1;    // SLS volume for image 1: H1 x W1 x dim (CV_32FC(dim))
    cv::Mat desc2;    // SLS volume for image 2
    cv::Mat pcaBasis; // optional PCA basis
};

SLSOutput extractScalelessDescs(const cv::Mat& I1,
    const cv::Mat& I2,
    bool usePaperParams);
