#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class SLSBuilder {
public:
    SLSBuilder(int subspaceDim);

    // Input: vector of H x W x p maps (PCA descriptors)
    // Output: H x W x D_sls map (CV_32FC(D_sls))
    cv::Mat buildSLSMap(const std::vector<cv::Mat>& mapsPCA) const;

private:
    int d_; // subspace dimension
};
