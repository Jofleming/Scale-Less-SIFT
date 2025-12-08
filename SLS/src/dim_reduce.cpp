#include "sls/dim_reduce.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

DimReduceResult dimReduce(const cv::Mat& dpMat1,
    const cv::Mat& dpMat2,
    const SLSOptions& opts)
{
    DimReduceResult out;

    if (opts.dimReduction <= 0)
    {
        out.dpMat1Reduced = dpMat1.clone();
        out.dpMat2Reduced = dpMat2.clone();
        return out;
    }

    int D = dpMat1.rows;
    int totalCols = opts.dimReductionCov;

    int N1 = dpMat1.cols;
    int N2 = dpMat2.cols;

    std::vector<int> idx1(N1), idx2(N2);
    for (int i = 0; i < N1; i++) idx1[i] = i;
    for (int i = 0; i < N2; i++) idx2[i] = i;

    cv::randShuffle(idx1);
    cv::randShuffle(idx2);

    if (totalCols > 0 && N1 + N2 > totalCols)
    {
        idx1.resize(totalCols / 2);
        idx2.resize(totalCols - (totalCols / 2));
    }

    cv::Mat samples;
    for (int j : idx1) samples.push_back(dpMat1.col(j).t());
    for (int j : idx2) samples.push_back(dpMat2.col(j).t());

    int reducedDim = opts.dimReduction;

    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW, reducedDim);

    cv::Mat eig = pca.eigenvectors.t();
    cv::Mat basis = eig.clone();

    out.pcaBasis = basis;

    out.dpMat1Reduced = basis.t() * dpMat1;
    out.dpMat2Reduced = basis.t() * dpMat2;

    return out;
}
