#include "sls/dense_sift.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

DescriptorGrid generateDescriptors(const cv::Mat& grayImage, const SLSOptions& opts) {
    DescriptorGrid out;

    if (grayImage.empty()) {
        std::cerr << "generateDescriptors: input image is empty\n";
        return out;
    }

    const int D = 128;
    const float NBP = 4.0f;

    int H = grayImage.rows;
    int W = grayImage.cols;

    out.s1 = (W + opts.gridSpacing - 1) / opts.gridSpacing;
    out.s2 = (H + opts.gridSpacing - 1) / opts.gridSpacing;
    out.numPoints = out.s1 * out.s2;

    const int numPoints = out.numPoints;
    const int numSigma = (int)opts.sigma.size();

    out.dpMat = cv::Mat::zeros(D, numPoints * numSigma, CV_32F);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    for (int si = 0; si < numSigma; ++si) {
        float sigma = opts.sigma[si];
        float SBP = 3.0f * sigma;
        int patchRadius = (int)std::round(SBP * (NBP + 1.0f) * 0.5f);
        int patchSize = 2 * patchRadius + 1;

        std::vector<cv::KeyPoint> kps;
        kps.reserve(numPoints);

        for (int y = 0; y < H; y += opts.gridSpacing) {
            for (int x = 0; x < W; x += opts.gridSpacing) {
                kps.emplace_back((float)x, (float)y, (float)patchSize);
            }
        }

        cv::Mat desc;
        sift->compute(grayImage, kps, desc);

        if (desc.empty()) {
            std::cerr << "[dense_sift] desc empty at sigma index " << si << "\n";
            continue;
        }

        for (int i = 0; i < desc.rows; ++i) {
            cv::Mat row = desc.row(i);
            double n = cv::norm(row, cv::NORM_L2);
            if (n > 1e-9) row /= n;
        }

        for (int i = 0; i < numPoints && i < desc.rows; ++i) {
            cv::Mat dstCol = out.dpMat.col(si + i * numSigma);
            desc.row(i).reshape(1, D).copyTo(dstCol);
        }
    }

    return out;
}
