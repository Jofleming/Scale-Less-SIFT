#include "sls/dim_reduce.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

DimReduceResult dimReduce(const cv::Mat& dpMat1,
    const cv::Mat& dpMat2,
    const SLSOptions& opts) {
    DimReduceResult out;

    if (opts.dimReduction == 0) {
        out.dpMat1Reduced = dpMat1.clone();
        out.dpMat2Reduced = dpMat2.clone();
        out.pcaBasis.release();
        return out;
    }

    int D = dpMat1.rows;
    int totalCols = opts.dimReductionCov;
    int colNum1 = dpMat1.cols;
    int colNum2 = dpMat2.cols;

    std::vector<int> cols1, cols2;

    if (totalCols > 0 && colNum1 + colNum2 > totalCols) {
        int keep1 = static_cast<int>(
            (double)colNum1 * totalCols / (colNum1 + colNum2)
            );
        int keep2 = totalCols - keep1;
        for (int i = 0; i < keep1; ++i) cols1.push_back(i);
        for (int i = 0; i < keep2; ++i) cols2.push_back(i);
    }
    else {
        for (int i = 0; i < colNum1; ++i) cols1.push_back(i);
        for (int i = 0; i < colNum2; ++i) cols2.push_back(i);
    }

    // stack samples as rows
    cv::Mat samples;
    for (int idx : cols1) {
        samples.push_back(dpMat1.col(idx).t());
    }
    for (int idx : cols2) {
        samples.push_back(dpMat2.col(idx).t());
    }

    int reducedDim = opts.dimReduction;

    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW, reducedDim);

    // eigenvectors: reducedDim x D; we want D x reducedDim
    cv::Mat eigVecsT = pca.eigenvectors.t();
    out.pcaBasis = eigVecsT.clone();

    // Project descriptors: reducedDim x cols = (reducedDim x D) * (D x cols)
    cv::Mat basisT = out.pcaBasis.t();
    out.dpMat1Reduced = basisT * dpMat1;
    out.dpMat2Reduced = basisT * dpMat2;

    return out;
}
