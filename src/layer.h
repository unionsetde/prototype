#include <stdio.h>
#include "activations.h"

struct network;
struct network_state;

typedef enum {
    CONVOLUTIONAL,
    SOFTMAX,
    COST,
    MAXPOOL
} LAYER_TYPE;

typedef enum{
    SSE
} COST_TYPE;

typedef struct layer {
    LAYER_TYPE type;
    int inputs;
    int outputs;
    int h, w, c;
    int size;
    int stride;
    int pad;
    int n;
    int out_h, out_w, out_c;
    float* weights;
    float* biases;
    ACTIVATION activation;

    void (*forward_gpu)   (struct layer, struct network_state);
    void (*backward_gpu)  (struct layer, struct network_state);
    void (*update_gpu)    (struct layer, int, float, float, float);

    float* x;
    float* output;
    float* delta;
    float* weight_updates;
    float* bias_updates;

    int batch;
    size_t workspace_size;
    int adam;
    float B1;
    float B2;
    float eps;
    float* m_gpu;
    float* v_gpu;
    int t;
    float* m;
    float* v;

    int batch_normalize;
    float* x_norm;
    float* scales;
    float* scale_updates;
    float* mean;
    float* variance;
    float* mean_delta;
    float* variance_delta;
    float* rolling_mean;
    float* rolling_variance;

    float* weights_gpu;
    float* weight_updates_gpu;
    float* biases_gpu;
    float* bias_updates_gpu;
    float* delta_gpu;
    float* output_gpu;

    float* mean_gpu;
    float* variance_gpu;
    float* rolling_mean_gpu;
    float* rolling_variance_gpu;
    float* mean_delta_gpu;
    float* variance_delta_gpu;
    float* scales_gpu;
    float* scale_updates_gpu;
    float* x_gpu;
    float* x_norm_gpu;

    // MAXPOOL
    int *indexes;
    int *indexes_gpu;
    // SOFTMAX
    int groups;
    float temperature;
    // COST
    COST_TYPE cost_type;
    int truths;
    float scale;
    float ratio;
    float* cost;
} layer;

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network {
    float *workspace;
    int n;
    int batch;
    int *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int h, w, c;
    int max_crop;
    int min_crop;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;

    int gpu_index;

    float **input_gpu;
    float **truth_gpu;
} network;

typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
} network_state;

typedef layer convolutional_layer;

convolutional_layer make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch, int batch_normalize);
void free_layer(layer l);
void forward_convolutional_layer_gpu(convolutional_layer l, network_state state);
void backward_convolutional_layer_gpu(convolutional_layer l, network_state state);
void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);
void pull_convolutional_layer(convolutional_layer layer);
void push_convolutional_layer(convolutional_layer layer);
void save_convolutional_weights(layer l, FILE *fp);
void load_convolutional_weights(layer l, FILE *fp);
// Supplementary functions
void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);
void forward_batchnorm_layer_gpu(layer l, network_state state);
void backward_batchnorm_layer_gpu(const layer l, network_state state);

typedef layer softmax_layer;
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network_state state);
void backward_softmax_layer_gpu(const softmax_layer l, network_state state);

typedef layer cost_layer;
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);
void forward_cost_layer_gpu(cost_layer l, network_state state);
void backward_cost_layer_gpu(const cost_layer l, network_state state);

typedef layer maxpool_layer;
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void forward_maxpool_layer_gpu(maxpool_layer l, network_state state);
void backward_maxpool_layer_gpu(maxpool_layer l, network_state state);
