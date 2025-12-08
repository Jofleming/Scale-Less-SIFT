#include "sls/sls_subspace.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

cv::Mat constructBasis(const cv::Mat& X, int subsDim) {
    int D = X.rows;
    int S = X.cols;

    cv::Mat mean;
    cv::reduce(X, mean, 1, cv::REDUCE_AVG);

    cv::Mat Xc = X.clone();
    for (int c = 0; c < S; ++c) {
        Xc.col(c) -= mean;
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(Xc, w, u, vt, cv::SVD::MODIFY_A);
    return u.colRange(0, subsDim).clone();
}

cv::Mat computeSLSDescriptors(const cv::Mat& dpMat,
    int numPoints,
    int s1, int s2,
    int numSigma,
    int subsDim) {
    int D = dpMat.rows;
    int numElements = D * (D + 1) / 2;

    cv::Mat Btemp(numElements, numPoints, CV_32F);

    std::vector<int> ind1;
    ind1.reserve(numElements);
    for (int r = 0; r < D; ++r) {
        for (int c = r; c < D; ++c) {
            int idx = r + c * D;
            ind1.push_back(idx);
        }
    }

    for (int i = 0; i < numPoints; ++i) {
        if (i % 1000 == 0) {
            std::cout << "SLS: pixel " << i << " / " << numPoints << std::endl;
        }

        int startCol = i * numSigma;
        int endCol = (i + 1) * numSigma;
        cv::Mat dp_i = dpMat(cv::Range::all(), cv::Range(startCol, endCol));

        cv::Mat B = constructBasis(dp_i, subsDim);

        cv::Mat A = B * B.t();

        cv::Mat diagMat = cv::Mat::zeros(D, D, CV_32F);
        for (int d = 0; d < D; ++d) {
            float v = A.at<float>(d, d);
            diagMat.at<float>(d, d) = v;
        }
        cv::Mat cMat = A - diagMat;
        cv::Mat b1 = diagMat * 0.5f;
        cv::Mat A2 = cMat + b1;

        // First transpose into a real Mat, then reshape
        cv::Mat AT = A2.t();
        AT = AT.reshape(1, D * D);

        cv::Mat h(numElements, 1, CV_32F);
        for (int k = 0; k < numElements; ++k) {
            h.at<float>(k, 0) = AT.at<float>(ind1[k], 0);
        }

        h.copyTo(Btemp.col(i));
    }

    cv::Mat sls(s2, s1, CV_32FC(numElements));

    for (int i = 0; i < numPoints; ++i) {
        int row = i / s1;
        int col = i % s1;
        float* dst = sls.ptr<float>(row, col);
        const float* src = Btemp.col(i).ptr<float>(0);
        std::memcpy(dst, src, numElements * sizeof(float));
    }

    return sls;
}
