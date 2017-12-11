#include <iostream>
#include <vector>
#include <string.h>

#include <numeric>

#include "ctc.h"

extern "C" int cpu_ctc(float *probs,
                        int prob_size, //probs->size[2]
                        float *grads,
                        int *labels,
                        int *label_sizes,
                        int *sizes,
                        int minibatch_size,
                        float *costs) {

    //std::cout << *probs << "|" << prob_size << "|" << *grads << "|" << *labels << "|" << *label_sizes << "|" << *sizes << "|" << minibatch_size << "|" << *costs << std::endl;
    //for (int i = 0; i < 10; i++) std::cout << probs[i] << " ";
    //std::cout << std::endl;
    //for (int i = 0; i < *label_sizes; i++) std::cout << labels[i] << " ";
    //std::cout << std::endl; 
    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes, sizes,
                       prob_size, minibatch_size,
                       options, &cpu_size_bytes);
    std::cout << cpu_size_bytes << std::endl;

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_ctc_loss(probs, grads,
                     labels, label_sizes,
                     sizes, prob_size,//probs->size[2],
                     minibatch_size, costs,
                     cpu_workspace, options);

    delete cpu_workspace;
    return 1;
}
