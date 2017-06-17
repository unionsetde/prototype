#include <stdlib.h>
#include <assert.h>

#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"

#include "layer.h"

convolutional_layer make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch, int batch_normalize)
{
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.h = h,
    l.w = w;
    l.c = c;
    l.size = size;
    l.n = n;
    l.stride = stride;
    l.pad = pad;
    l.batch = batch;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    int cnt;
    float scale = sqrt(2./(size*size*c));
    for (cnt = 0; cnt < c*n*size*size; ++cnt) l.weights[cnt] = scale*rand_uniform(-1, 1);

    int out_h = (h + 2*pad - size) / stride + 1;
    int out_w = (w + 2*pad - size) / stride + 1;
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta = calloc(l.batch*l.outputs, sizeof(float));

    if (batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for (cnt = 0; cnt < n; ++cnt){
            l.scales[cnt] = 1;
        }
        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));

        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    // ADAM optimizer
    if (l.adam){ // false all the time
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.m_gpu = cuda_make_array(l.m, c*n*size*size);
        l.v_gpu = cuda_make_array(l.v, c*n*size*size);
    }

    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

    if (batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, n);
        l.variance_gpu = cuda_make_array(l.variance, n);

        l.rolling_mean_gpu = cuda_make_array(l.mean, n);
        l.rolling_variance_gpu = cuda_make_array(l.variance, n);

        l.mean_delta_gpu = cuda_make_array(l.mean, n);
        l.variance_delta_gpu = cuda_make_array(l.variance, n);

        l.scales_gpu = cuda_make_array(l.scales, n);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

        l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    }

    l.activation = activation;
    l.workspace_size = (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void free_layer(layer l)
{
    if (l.weights) free(l.weights);
    if (l.biases) free(l.biases);

    if (l.x) free(l.x);
    if (l.output) free(l.output);
    if (l.delta) free(l.delta);
    if (l.weight_updates) free(l.weight_updates);
    if (l.bias_updates) free(l.bias_updates);

    if (l.m) free(l.m);
    if (l.v) free(l.v);

    if (l.x_norm) free(l.x_norm);
    if (l.scales) free(l.scales);
    if (l.scale_updates) free(l.scale_updates);
    if (l.mean) free(l.mean);
    if (l.variance) free(l.variance);
    if (l.mean_delta) free(l.mean_delta);
    if (l.variance_delta) free(l.variance_delta);
    if (l.rolling_mean) free(l.rolling_mean);
    if (l.rolling_variance) free(l.rolling_variance);

    if (l.m_gpu) cuda_free(l.m_gpu);
    if (l.v_gpu) cuda_free(l.v_gpu);
    if (l.weights_gpu) cuda_free(l.weights_gpu);
    if (l.weight_updates_gpu) cuda_free(l.weight_updates_gpu);
    if (l.biases_gpu) cuda_free(l.biases_gpu);
    if (l.bias_updates_gpu) cuda_free(l.bias_updates_gpu);
    if (l.delta_gpu) cuda_free(l.delta_gpu);
    if (l.output_gpu) cuda_free(l.output_gpu);

    if (l.mean_gpu) cuda_free(l.mean_gpu);
    if (l.variance_gpu) cuda_free(l.variance_gpu);
    if (l.rolling_mean_gpu) cuda_free(l.rolling_mean_gpu);
    if (l.rolling_variance_gpu) cuda_free(l.rolling_variance_gpu);
    if (l.mean_delta_gpu) cuda_free(l.mean_delta_gpu);
    if (l.variance_delta_gpu) cuda_free(l.variance_delta_gpu);
    if (l.scales_gpu) cuda_free(l.scales_gpu);
    if (l.scale_updates_gpu) cuda_free(l.scale_updates_gpu);
    if (l.x_gpu) cuda_free(l.x_gpu);
    if (l.x_norm_gpu) cuda_free(l.x_norm_gpu);

    if (l.cost) free(l.cost); // for cost layer
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        float * a = l.weights_gpu;
        float * b = state.workspace;
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }
    if (l.batch_normalize){
        forward_batchnorm_layer_gpu(l, state);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    if (l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    }
    else{
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.output_gpu, 1, l.delta_gpu, 1);
    }
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.weight_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);
        if (state.delta){
            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
        }
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    if (layer.scales_gpu){
        axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
        scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
    }

    if (layer.adam){
        scal_ongpu(size, layer.B1, layer.m_gpu, 1);
        scal_ongpu(size, layer.B2, layer.v_gpu, 1);

        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

        axpy_ongpu(size, -(1-layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
        mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, (1-layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

        adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate/batch, layer.eps, layer.t+1);
        fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
    }
    else{
        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate/batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
        scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
    }
}

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

void forward_batchnorm_layer_gpu(layer l, network_state state)
{
    if (state.train){
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_ongpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_ongpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
    }
    else{
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    }

    scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer_gpu(const layer l, network_state state)
{
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
}

void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    pull_convolutional_layer(l);
    int num = l.n*l.c*l.size*l.size;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
    if (l.adam){
        fwrite(l.m, sizeof(float), num, fp);
        fwrite(l.v, sizeof(float), num, fp);
    }
}

void load_convolutional_weights(layer l, FILE *fp)
{
    int num = l.n*l.c*l.size*l.size;
    size_t status = 1;
    status *= fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        status *= fread(l.scales, sizeof(float), l.n, fp);
        status *= fread(l.rolling_mean, sizeof(float), l.n, fp);
        status *= fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    status *= fread(l.weights, sizeof(float), num, fp);
    if (l.adam){
        status *= fread(l.m, sizeof(float), num, fp);
        status *= fread(l.v, sizeof(float), num, fp);
    }
    if (status == 0) printf("Error with fread in load_convolutional_weights...\n");
    push_convolutional_layer(l);
}

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);

    return l;
}

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network_state state)
{
    int inputs = l.inputs / l.groups;
    int batch = l.batch * l.groups;

    softmax_gpu(state.input, inputs, inputs, batch, l.temperature, l.output_gpu);
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}

cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "cost                                           %4d\n",  inputs);
    cost_layer l = {0};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward_gpu = forward_cost_layer_gpu;
    l.backward_gpu = backward_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);

    return l;
}

void resize_cost_layer(cost_layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = realloc(l->output, inputs*l->batch*sizeof(float));
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
}

void pull_cost_layer(cost_layer l)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_cost_layer(cost_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if (fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if (fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu(cost_layer l, network_state state)
{
    if (!state.truth) return;
    l2_gpu(l.batch*l.inputs, state.input, state.truth, l.delta_gpu, l.output_gpu);
    if (l.ratio){
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        supp_ongpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer_gpu(const cost_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, state.delta, 1);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);

    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}
