#pragma once
#include <opencv2/opencv.hpp>

class DenseMatcher {
public:
    DenseMatcher(int searchRadius);

    // desc1, desc2: H x W x D (CV_32FC(D))
    // returns flow: H x W x 2 (CV_32FC2) with (dx, dy)
    cv::Mat match(const cv::Mat& desc1, const cv::Mat& desc2) const;

private:
    int r_;
};
