#pragma once
#include <vector>

struct SLSOptions {
    std::vector<float> sigma;
    int dimReduction;
    int dimReductionCov;
    int subsDim;
    int gridSpacing;

    SLSOptions()
        : dimReduction(32),
        dimReductionCov(50000),
        subsDim(10),
        gridSpacing(1)
    {
    }
};

