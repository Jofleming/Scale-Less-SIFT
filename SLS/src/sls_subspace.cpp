#include "sls/sls_subspace.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>

cv::Mat computeSLSDescriptors(const cv::Mat& dpMat,
    int numPoints,
    int s1, int s2,
    int numSigma,
    int subsDim)
{
    const int D = dpMat.rows;
    const int SL = D * (D + 1) / 2;

    cv::Mat sls(s2, s1, CV_32FC(SL));

    std::vector<int> inds;
    inds.reserve(SL);
    for (int i = 0; i < D; ++i)
        for (int j = i; j < D; ++j)
            inds.push_back(i * D + j);

    const float diagScale = 1.0f / std::sqrt(2.0f);

    for (int p = 0; p < numPoints; ++p)
    {
        int start = p * numSigma;
        int end = start + numSigma;
        if (end > dpMat.cols) end = dpMat.cols;
        int K = end - start;
        if (K <= 0) continue;

        cv::Mat X = dpMat(cv::Range::all(), cv::Range(start, end)).clone();

        cv::Mat mean;
        cv::reduce(X, mean, 1, cv::REDUCE_AVG);
        for (int c = 0; c < X.cols; ++c)
            X.col(c) -= mean;

        cv::Mat W, U, VT;
        cv::SVD::compute(X, W, U, VT, cv::SVD::MODIFY_A);

        int d = std::min(std::max(1, subsDim), D);
        cv::Mat H = U.colRange(0, d).clone();

        cv::Mat A = H * H.t();

        for (int i = 0; i < D; ++i)
            A.at<float>(i, i) *= diagScale;

        cv::Mat flat = A.reshape(1, D * D);

        cv::Mat h(1, SL, CV_32F);
        for (int k = 0; k < SL; ++k)
            h.at<float>(0, k) = flat.at<float>(inds[k], 0);

        float nrm = cv::norm(h);
        if (nrm > 1e-8f) h /= nrm;

        int row = p / s1;
        int col = p % s1;
        memcpy(sls.ptr<float>(row, col), h.ptr<float>(), SL * sizeof(float));
    }

    return sls;
}
