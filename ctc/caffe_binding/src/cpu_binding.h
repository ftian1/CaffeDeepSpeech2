int cpu_ctc(float *probs, int prob_size,
                        float *grads,
                        int *labels,
                        int *label_sizes,
                        int *sizes,
                        int minibatch_size,
                        float *costs);
