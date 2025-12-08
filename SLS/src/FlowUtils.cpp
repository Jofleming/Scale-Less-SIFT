#include "sls/FlowUtils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace sls {

    cv::Mat computeDenseFlowLocal(
        const cv::Mat& sourceDesc,
        const cv::Mat& targetDesc,
        int windowRadius)
    {
        CV_Assert(sourceDesc.size() == targetDesc.size());
        CV_Assert(sourceDesc.type() == targetDesc.type());
        CV_Assert(sourceDesc.depth() == CV_32F);

        int H = sourceDesc.rows;
        int W = sourceDesc.cols;
        int C = sourceDesc.channels();

        cv::Mat flow(H, W, CV_32FC2);

        for (int y = 0; y < H; ++y) {
            // simple progress indicator
            if (y % 10 == 0) {
                std::cout << "computeDenseFlowLocal: row " << y << " / " << H << "\r";
                std::cout.flush();
            }
            for (int x = 0; x < W; ++x) {
                const float* fs = sourceDesc.ptr<float>(y, x);

                float bestDist = std::numeric_limits<float>::max();
                int bestX = x;
                int bestY = y;

                int y0 = std::max(0, y - windowRadius);
                int y1 = std::min(H - 1, y + windowRadius);
                int x0 = std::max(0, x - windowRadius);
                int x1 = std::min(W - 1, x + windowRadius);

                for (int yy = y0; yy <= y1; ++yy) {
                    for (int xx = x0; xx <= x1; ++xx) {
                        const float* ft = targetDesc.ptr<float>(yy, xx);
                        float dist = 0.0f;

                        for (int c = 0; c < C; ++c) {
                            float d = fs[c] - ft[c];
                            dist += d * d;
                        }
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestX = xx;
                            bestY = yy;
                        }
                    }
                }

                cv::Vec2f& v = flow.at<cv::Vec2f>(y, x);
                v[0] = static_cast<float>(bestX - x);
                v[1] = static_cast<float>(bestY - y);
            }
        }
        std::cout << std::endl;
        return flow;
    }

    cv::Mat flowToColor(const cv::Mat& flow)
    {
        CV_Assert(flow.type() == CV_32FC2);
        cv::Mat hsv(flow.size(), CV_32FC3);

        for (int y = 0; y < flow.rows; ++y) {
            for (int x = 0; x < flow.cols; ++x) {
                cv::Vec2f v = flow.at<cv::Vec2f>(y, x);
                float angle = std::atan2(v[1], v[0]);
                float mag = std::sqrt(v[0] * v[0] + v[1] * v[1]);

                float h = (angle + CV_PI) * (180.0f / CV_PI) / 2.0f;
                float s = 1.0f;
                float vval = std::min(mag * 0.1f, 1.0f);

                hsv.at<cv::Vec3f>(y, x) = cv::Vec3f(h, s, vval);
            }
        }

        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        return bgr;
    }

    cv::Mat warpImage(const cv::Mat& target, const cv::Mat& flow)
    {
        CV_Assert(target.rows == flow.rows && target.cols == flow.cols);
        cv::Mat mapX(target.size(), CV_32F);
        cv::Mat mapY(target.size(), CV_32F);

        for (int y = 0; y < target.rows; ++y) {
            for (int x = 0; x < target.cols; ++x) {
                cv::Vec2f v = flow.at<cv::Vec2f>(y, x);
                mapX.at<float>(y, x) = static_cast<float>(x) + v[0];
                mapY.at<float>(y, x) = static_cast<float>(y) + v[1];
            }
        }

        cv::Mat warped;
        cv::remap(target, warped, mapX, mapY, cv::INTER_LINEAR);
        return warped;
    }

    FlowEvalResult evaluateFlowAgainstHomography(
        const cv::Mat& flow,
        const cv::Mat& H,
        int step)
    {
        CV_Assert(flow.type() == CV_32FC2);
        CV_Assert(H.rows == 3 && H.cols == 3);
        CV_Assert(H.type() == CV_64F);

        int Himg = flow.rows;
        int Wimg = flow.cols;

        cv::Matx33d Hm;
        H.copyTo(Hm);

        std::vector<double> errors;
        errors.reserve((Himg / step) * (Wimg / step));

        for (int y = 0; y < Himg; y += step) {
            for (int x = 0; x < Wimg; x += step) {
                cv::Vec2f v = flow.at<cv::Vec2f>(y, x);
                double xPred = static_cast<double>(x) + v[0];
                double yPred = static_cast<double>(y) + v[1];

                cv::Vec3d p(static_cast<double>(x),
                    static_cast<double>(y),
                    1.0);
                cv::Vec3d q = Hm * p;
                double u = q[0] / q[2];
                double vgt = q[1] / q[2];

                double dx = xPred - u;
                double dy = yPred - vgt;
                double err = std::sqrt(dx * dx + dy * dy);
                errors.push_back(err);
            }
        }

        FlowEvalResult res{};
        res.numSamples = static_cast<int>(errors.size());
        if (errors.empty()) {
            res.meanError = res.medianError = 0.0;
            res.percentBelow2px = res.percentBelow5px = 0.0;
            return res;
        }

        double sum = 0.0;
        for (double e : errors) sum += e;
        res.meanError = sum / errors.size();

        std::sort(errors.begin(), errors.end());
        if (errors.size() % 2 == 1) {
            res.medianError = errors[errors.size() / 2];
        }
        else {
            res.medianError = 0.5 * (errors[errors.size() / 2 - 1] +
                errors[errors.size() / 2]);
        }

        int count2 = 0, count5 = 0;
        for (double e : errors) {
            if (e <= 2.0) ++count2;
            if (e <= 5.0) ++count5;
        }
        res.percentBelow2px = 100.0 * (double)count2 / errors.size();
        res.percentBelow5px = 100.0 * (double)count5 / errors.size();

        return res;
    }

}
