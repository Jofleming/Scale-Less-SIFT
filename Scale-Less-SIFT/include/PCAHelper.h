#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PCAHelper {
public:
    PCAHelper();

    // Collect sample descriptors from dense maps
    void collectSamples(const std::vector<cv::Mat>& maps, int maxSamples);

    // Perform PCA to reduce to targetDim
    void computePCA(int targetDim);

    // Project a single descriptor row (1 x inputDim)
    void projectRow(const cv::Mat& srcRow, cv::Mat& dstRow) const;

    // Project all dense maps
    std::vector<cv::Mat> projectMaps(const std::vector<cv::Mat>& maps) const;

    int inputDim() const { return inputDim_; }
    int outputDim() const { return outputDim_; }

private:
    cv::Mat samples_; // N x inputDim
    cv::PCA pca_;
    bool trained_;
    int inputDim_;
    int outputDim_;
};
