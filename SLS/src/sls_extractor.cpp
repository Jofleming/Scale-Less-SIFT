#include "sls/sls_extractor.hpp"
#include "sls/dense_sift.hpp"
#include "sls/dim_reduce.hpp"
#include "sls/sls_subspace.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

SLSOutput extractScalelessDescs(const cv::Mat& I1,
    const cv::Mat& I2,
    bool usePaperParams)
{
    SLSOptions opts;

    int numSigma = 20;
    opts.sigma.clear();
    float minS = 0.5f, maxS = 12.0f;

    for (int i = 0; i < numSigma; ++i)
    {
        float t = (float)i / (numSigma - 1);
        float s = minS * std::pow(maxS / minS, t);
        opts.sigma.push_back(s);
    }

    opts.gridSpacing = 4;

    if (usePaperParams) {
        opts.dimReduction = 30;
        opts.dimReductionCov = 50000;
        opts.subsDim = std::min(8, opts.dimReduction);
    }
    else {
        opts.dimReduction = 30;
        opts.dimReductionCov = 50000;
        opts.subsDim = std::min(10, opts.dimReduction);
    }

    cv::Mat g1, g2;
    if (I1.channels() == 3) cv::cvtColor(I1, g1, cv::COLOR_BGR2GRAY);
    else g1 = I1.clone();

    if (I2.channels() == 3) cv::cvtColor(I2, g2, cv::COLOR_BGR2GRAY);
    else g2 = I2.clone();

    g1.convertTo(g1, CV_8U);
    g2.convertTo(g2, CV_8U);

    DescriptorGrid grid1 = generateDescriptors(g1, opts);
    DescriptorGrid grid2 = generateDescriptors(g2, opts);

    DimReduceResult dr = dimReduce(grid1.dpMat, grid2.dpMat, opts);

    cv::Mat dp1 = dr.dpMat1Reduced;
    cv::Mat dp2 = dr.dpMat2Reduced;

    cv::Mat sls1 = computeSLSDescriptors(dp1,
        grid1.numPoints,
        grid1.s1, grid1.s2,
        numSigma,
        opts.subsDim);

    cv::Mat sls2 = computeSLSDescriptors(dp2,
        grid2.numPoints,
        grid2.s1, grid2.s2,
        numSigma,
        opts.subsDim);

    SLSOutput out;
    out.desc1 = sls1;
    out.desc2 = sls2;
    out.pcaBasis = dr.pcaBasis;

    return out;
}
