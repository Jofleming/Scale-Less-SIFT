#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "DenseSIFT.h"

class MultiScaleSIFTExtractor {
public:
    MultiScaleSIFTExtractor(int numScales, float sigmaMin, float sigmaMax, int step);

    // Input: grayscale 8-bit image
    // Output: vector of H x W x 128 (CV_32FC(128)), one per scale
    std::vector<cv::Mat> compute(const cv::Mat& gray);

private:
    int numScales_;
    float sigmaMin_;
    float sigmaMax_;
    DenseSIFT denseSift_;
};
