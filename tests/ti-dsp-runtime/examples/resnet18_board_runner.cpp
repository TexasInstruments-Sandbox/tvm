/*
 * resnet18_board_runner.cpp -- ResNet-18 image classification, on the board.
 *
 * This is the board-side C++ API example: load the compiled DLOAD module,
 * run one preprocessed image through it, and print the top-5 ImageNet
 * class predictions. Quantizing, compiling, and preprocessing the image
 * all happen on the dev host (see run_resnet18_classification.py, which
 * deploys and invokes this binary over SSH); the only work done here is
 * the inference call itself and the argmax + label lookup -- nothing gets
 * shipped back to the host as raw tensors.
 *
 * This is the direct C++ analogue of yolo26_board_runner.py. c7x_infer.h
 * is the "common inference code" shared by every board-side example; this
 * file is the task-specific "main application" -- a future object
 * detection example would #include the same header but decode boxes here
 * instead of computing an argmax.
 *
 * Usage (normally invoked by run_resnet18_classification.py over SSH; can
 * also be run by hand on the board for debugging):
 *
 *   resnet18_board_runner lib0.out input.bin labels.txt
 *
 * input.bin must be a raw flat float32 [1,3,224,224] tensor (no header),
 * already resized and ImageNet-normalized -- see
 * run_resnet18_classification.py for how it's produced. labels.txt is one
 * ImageNet class name per line, in output-index order.
 */
#include "c7x_infer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using c7x_examples::InferenceSession;
using c7x_examples::ParseDType;
using c7x_examples::ReadRawTensor;

namespace {

const std::vector<int64_t> kInputShape = {1, 3, 224, 224};

std::vector<std::string> ReadLabels(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(f, line)) labels.push_back(line);
    return labels;
}

} /* namespace */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <lib0.out> <input.bin> <labels.txt>\n", argv[0]);
        return 1;
    }
    const std::string lib0_path = argv[1];
    const std::string input_path = argv[2];
    const std::string labels_path = argv[3];

    try {
        auto input = ReadRawTensor(input_path);
        auto labels = ReadLabels(labels_path);

        InferenceSession session(lib0_path);
        auto out = session.Run(input, kInputShape, ParseDType("float32"));

        /* This model's output is dequantized back to float32 -- verify
         * that before reinterpreting out.dl.data as float*, since a
         * still-quantized (int8/int16) output would otherwise silently
         * produce garbage logits with no error. */
        if (out.dl.dtype.code != 2 || out.dl.dtype.bits != 32) {
            fprintf(stderr, "ERROR: expected float32 output, got dtype code=%d bits=%d\n",
                    out.dl.dtype.code, out.dl.dtype.bits);
            return 1;
        }
        const float *logits = static_cast<const float *>(out.dl.data);
        size_t num_classes = out.data_size / sizeof(float);
        if (num_classes != labels.size()) {
            fprintf(stderr, "WARNING: %zu output classes but %zu labels\n", num_classes,
                    labels.size());
        }

        std::vector<size_t> order(num_classes);
        for (size_t i = 0; i < num_classes; ++i) order[i] = i;
        size_t top_k = std::min<size_t>(5, num_classes);
        /* logits[a] > logits[b] alone is undefined behavior under
         * strict-weak-ordering if either is NaN (a bad inference result,
         * not expected in practice, but worth not compounding with UB) --
         * sort NaNs to the bottom instead of comparing them. */
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                          [&](size_t a, size_t b) {
                              if (std::isnan(logits[a])) return false;
                              if (std::isnan(logits[b])) return true;
                              return logits[a] > logits[b];
                          });

        printf("Top-%zu predictions:\n", top_k);
        for (size_t i = 0; i < top_k; ++i) {
            size_t idx = order[i];
            const char *label = idx < labels.size() ? labels[idx].c_str() : "?";
            printf("  %-30s logit=%.4f\n", label, logits[idx]);
        }
        session.Close();
    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
