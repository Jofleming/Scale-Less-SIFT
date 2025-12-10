#include "sls/sls_extractor.hpp"
#include "sls/sls_options.hpp"
#include "sls/dense_sift.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;

// Perform PCA on stacked samples and project both descriptor matrices.
// dp1, dp2: D x N matrices (columns are descriptors).
// If opts.dimReduction <= 0, no PCA is done and they are returned unchanged.
static void pcaReduce(const Mat& dp1,
    const Mat& dp2,
    const SLSOptions& opts,
    Mat& dp1Reduced,
    Mat& dp2Reduced,
    Mat& pcaBasis)
{
    const int D = dp1.rows;

    if (opts.dimReduction <= 0 || opts.dimReduction >= D) {
        dp1Reduced = dp1.clone();
        dp2Reduced = dp2.clone();
        pcaBasis = Mat::eye(D, D, CV_32F);
        std::cout << "[SLS] PCA disabled (using original descriptor dimension " << D << ").\n";
        return;
    }

    // Stack all descriptors from both images as row samples.
    Mat dp1T = dp1.t();
    Mat dp2T = dp2.t();
    Mat all;
    vconcat(dp1T, dp2T, all);

    std::cout << "[SLS] Running PCA with target dim = " << opts.dimReduction
        << " on " << all.rows << " samples of dim " << all.cols << "...\n";

    PCA pca(all, Mat(), PCA::DATA_AS_ROW, opts.dimReduction);

    // Project both sets and transpose back to D' x N.
    Mat proj1, proj2;
    pca.project(dp1T, proj1);
    pca.project(dp2T, proj2);

    Mat proj1T = proj1.t();
    Mat proj2T = proj2.t();
    dp1Reduced = proj1T.clone();
    dp2Reduced = proj2T.clone();

    pcaBasis = pca.eigenvectors.clone();

    std::cout << "[SLS] PCA done. New descriptor dimension = "
        << dp1Reduced.rows << ".\n";
}


// Average descriptors across scales for each pixel.
// dp: D x (numPoints * numSigma), column layout: si + i * numSigma
// Returns: D x numPoints matrix (one descriptor per pixel).
static Mat averageAcrossScales(const Mat& dp,
    int numPoints,
    int numSigma)
{
    CV_Assert(dp.cols == numPoints * numSigma);
    const int D = dp.rows;

    Mat desc(D, numPoints, CV_32F);
    desc.setTo(0);

    float invNumSigma = 1.0f / static_cast<float>(numSigma);

    for (int i = 0; i < numPoints; ++i) {
        Mat outCol = desc.col(i);
        outCol.setTo(0);

        for (int s = 0; s < numSigma; ++s) {
            int colIdx = s + i * numSigma;
            outCol += dp.col(colIdx);
        }
        outCol *= invNumSigma;
    }

    return desc;
}

// Extract SLS-like descriptors for two images.
SLSOutput extractScalelessDescs(const Mat& I1, const Mat& I2, bool usePaperParams)
{
    SLSOutput out;

    if (I1.empty() || I2.empty()) {
        std::cerr << "extractScalelessDescs: one of the input images is empty.\n";
        return out;
    }

    SLSOptions opts;

    if (usePaperParams) {
        opts.sigma.clear();
        int numSigma = 20;
        float sigmaStart = 0.5f;
        float sigmaEnd = 12.0f;
        float step = (sigmaEnd - sigmaStart) / (numSigma - 1);
        for (int i = 0; i < numSigma; ++i) {
            opts.sigma.push_back(sigmaStart + i * step);
        }

        opts.gridSpacing = 1;
        opts.dimReduction = 0;
        opts.dimReductionCov = 50000;
        opts.subsDim = 8;

        std::cout << "[SLS] Using paper-like parameters (dense, many scales).\n";
    }
    else {
        // Lightweight parameters for debugging / development.
        opts.sigma.clear();
        int numSigma = 3;
        float sigmaStart = 1.0f;
        float sigmaEnd = 4.0f;
        float step = (sigmaEnd - sigmaStart) / (numSigma - 1);
        for (int i = 0; i < numSigma; ++i) {
            opts.sigma.push_back(sigmaStart + i * step);
        }

        opts.gridSpacing = 8;
        opts.dimReduction = 32;
        opts.dimReductionCov = 20000;
        opts.subsDim = 6;

        std::cout << "[SLS] Using lightweight SLS parameters.\n";
    }

    // Convert to grayscale float [0,1]
    Mat g1, g2;

    if (I1.channels() > 1) {
        cvtColor(I1, g1, COLOR_BGR2GRAY);
    }
    else {
        g1 = I1.clone();
    }

    if (I2.channels() > 1) {
        cvtColor(I2, g2, COLOR_BGR2GRAY);
    }
    else {
        g2 = I2.clone();
    }

    g1.convertTo(g1, CV_32F, 1.0 / 255.0);
    g2.convertTo(g2, CV_32F, 1.0 / 255.0);

    // --- Dense SIFT descriptors at multiple scales ---
    std::cout << "[SLS] Generating dense descriptors for image 1...\n";
    DescriptorGrid dp1 = generateDescriptors(g1, opts);
    std::cout << "[SLS] ...image 1 descriptors done.\n";

    std::cout << "[SLS] Generating dense descriptors for image 2...\n";
    DescriptorGrid dp2 = generateDescriptors(g2, opts);
    std::cout << "[SLS] ...image 2 descriptors done.\n";

    // PCA / dimensionality reduction
    Mat dp1Reduced, dp2Reduced, pcaBasis;
    pcaReduce(dp1.dpMat, dp2.dpMat, opts, dp1Reduced, dp2Reduced, pcaBasis);

    const int numSigma = static_cast<int>(opts.sigma.size());

    // Build one descriptor per pixel by averaging across scales
    std::cout << "[SLS] Averaging descriptors across scales for image 1...\n";
    Mat desc1 = averageAcrossScales(dp1Reduced, dp1.numPoints, numSigma);

    std::cout << "[SLS] Averaging descriptors across scales for image 2...\n";
    Mat desc2 = averageAcrossScales(dp2Reduced, dp2.numPoints, numSigma);

    out.desc1 = desc1;
    out.desc2 = desc2;
    out.pcaBasis = pcaBasis;

    std::cout << "[SLS] Finished extractScalelessDescs.\n";
    return out;
}
