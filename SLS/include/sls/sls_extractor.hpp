#pragma once
#include <opencv2/core.hpp>
#include "sls_options.hpp"

struct SLSOutput {
    cv::Mat desc1;
    cv::Mat desc2;
    cv::Mat pcaBasis;
};

SLSOutput extractScalelessDescs(const cv::Mat& I1,
    const cv::Mat& I2,
    bool usePaperParams);
