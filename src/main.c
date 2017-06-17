#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "curand.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include "layer.h"

typedef struct {
    int h;
    int w;
    int c;
    float *data;
} image;

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

void free_image(image m)
{
    if (m.data){
        free(m.data);
    }
}

image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count=0;

    for (k = 0; k < c; ++k){
        for (i = 0; i < h; ++i){
            for (j = 0; j < w; ++j){
                out.data[count++] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return out;
}

void rgbgr_image(image im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image load_image_cv(char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if ((src = cvLoadImage(filename, flag)) == 0)
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        //char buff[256];
        //sprintf(buff, "echo %s >> bad.list", filename);
        //system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    rgbgr_image(out);
    return out;
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

void show_image_cv(image p, const char *name)
{
    int x,y,k;
    image copy = copy_image(p);
    if (p.c == 3) rgbgr_image(copy);

    char buff[256];
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL);
    for (y = 0; y < p.h; ++y){
        for (x = 0; x < p.w; ++x){
            for (k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    free_image(copy);
    cvNamedWindow(buff,CV_WINDOW_NORMAL);
    cvShowImage(buff, disp);
    cvWaitKey(0);
    cvReleaseImage(&disp);
}

image load_image(char *filename, int w, int h, int c)
{
    image out = load_image_cv(filename, c);

    if ((h && w) && (h != out.h || w != out.w)){
        printf("image from %s does not fit (%d, %d, %d)\n", filename, w, h, c);
    }
    return out;
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
    return net;
}

void free_network(network net)
{
    int i;
    for (i = 0; i < net.n; ++i){
        free_layer(net.layers[i]);
    }
    free(net.layers);
    if (*net.input_gpu) cuda_free(*net.input_gpu);
    if (*net.truth_gpu) cuda_free(*net.truth_gpu);
    if (net.input_gpu) free(net.input_gpu);
    if (net.truth_gpu) free(net.truth_gpu);
}

network create_network(void)
{
    int n = 8;
    network net = make_network(n);
    net.gpu_index = 0;

    net.batch = 128;
    net.subdivisions = 1;
    net.learning_rate=0.1;
    net.policy = POLY;
    net.power=4;
    net.max_batches = 5000;
    net.momentum = 0.9;
    net.decay = 0.0005;

    net.h = 32;
    net.w = 32;
    net.c = 3;
    net.inputs = 32*32*3;
    size_t workspace_size = 0;
    int batch=net.batch, batch_normalize=1;
    int h, w, c, nf, size, stride, pad;
    ACTIVATION activation = LEAKY;

    // initialize layers
    h=32; w=32; c=3; nf=32; size=3; stride=1; pad=1;
    activation = LEAKY;
    layer l1 = {0};
    l1 = make_convolutional_layer(h,w,c,nf,size,stride,pad,activation,batch,batch_normalize);
    net.layers[0] = l1;
    if (l1.workspace_size > workspace_size) workspace_size = l1.workspace_size;

    layer l2 = {0};
    l2 = make_maxpool_layer(batch,l1.out_h,l1.out_w,l1.out_c,2,2,1/2);
    net.layers[1] = l2;

    h=l2.out_h; w=l2.out_w; c=nf; nf=64; size=3; stride=1; pad=1;
    activation = LEAKY;
    layer l3 = {0};
    l3 = make_convolutional_layer(h,w,c,nf,size,stride,pad,activation,batch,batch_normalize);
    net.layers[2] = l3;
    if (l3.workspace_size > workspace_size) workspace_size = l3.workspace_size;

    layer l4 = {0};
    l4 = make_maxpool_layer(batch,h,w,nf,2,2,1/2);
    net.layers[3] = l4;

    h=l4.out_h; w=l4.out_w; c=l4.out_c; nf=128; size=3; stride=1; pad=1;
    activation = LEAKY;
    layer l5 = {0};
    l5 = make_convolutional_layer(h,w,c,nf,size,stride,pad,activation,batch,batch_normalize);
    net.layers[4] = l5;
    if (l5.workspace_size > workspace_size) workspace_size = l5.workspace_size;

    // fully_connected_layer configuration
    h=l5.out_h; w=l5.out_w; c=l5.out_c; nf=10; size=l5.out_w; stride=l5.out_h; pad=0;
    activation = LINEAR;
    layer l6 = {0};
    l6 = make_convolutional_layer(h,w,c,nf,size,stride,pad,activation,batch,batch_normalize);
    net.layers[5] = l6;
    if (l6.workspace_size > workspace_size) workspace_size = l6.workspace_size;

    layer l8 = {0};
    l8 = make_softmax_layer(batch,10,1);
    l8.temperature = 1;
    net.layers[6] = l8;

    layer l9 = {0};
    l9 = make_cost_layer(batch, 10, SSE, 1);
    l9.ratio = 0;
    net.layers[7] = l9;
    //net.outputs = net.layers[net.n-1].outputs;
    //net.output = get_network_output(net);
    if (workspace_size){
        if (gpu_index >= 0){
            net.workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
        else{
            net.workspace = calloc(1, workspace_size);
        }
    }
    return net;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

void forward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if (l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, state);
        state.input = l.output_gpu;
    }
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for (i = net.n-1; i > 0; --i) if (net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    cuda_set_device(net.gpu_index);
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
}

int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    switch (net.policy){
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for (i = 0; i < net.num_steps; ++i){
                if (net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if (net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

void update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for (i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        if (l.update_gpu){
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

int get_network_output_size(network net)
{
    int i;
    for (i = net.n-1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

void backward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for (i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if (i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }
        else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        l.backward_gpu(l, state);
    }
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if (net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if (!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }
    else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    backward_network_gpu(net, state);
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for (i = 0; i < net.n; ++i){
        if (net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    return sum/count;
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    *net.seen += net.batch;
    forward_backward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);
    return error;
}

void test_forward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if (net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if (!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }
    else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 0;
    forward_network_gpu(net, state);
}

float test_network_datum_gpu(network net, float *x, float *y)
{
    *net.seen += net.batch;
    test_forward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    return error;
}

typedef struct cifa_data {
    image icon;
    int label;
} cifa_data;

void free_cifa_data(cifa_data sample)
{
    if (sample.icon.data) free_image(sample.icon);
}

int get_cifar_label(char* filename)
{
    int cnt = 0;
    char buff[256];
    int label = -1;
    const char* delim = "/_.";
    const char *labels[] = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};

    strcpy(buff, filename);
    char* token = strtok(buff, delim);

    while (token){
        for (cnt = 0; cnt < 10; cnt++){
            if (0 == strcmp(token, labels[cnt])){
                label = cnt;
                break;
            }
        }
        if (label != -1) break;
        token = strtok(NULL, delim);
    }
    if (-1 == label) printf("No label in filename: %s\n", filename);
    return label;
}

cifa_data fetch_cifa_data(char* filename)
{
    cifa_data data = {{0}};
    data.icon = load_image(filename,32,32,3);
    data.label = get_cifar_label(filename);
    return data;
}

float* get_cifa_output(cifa_data data, int out_c, int out_h, int out_w)
{
    float* output = calloc(out_c*out_h*out_w, sizeof(float));
    output[data.label] = 1.0;
    return output;
}

typedef struct batch_data {
    int batch;
    float* data_icon;
    float* data_truth;
    int* label;
} batch_data;

batch_data make_batch_data(int batch)
{
    batch_data raw = {0};

    raw.batch = 0;
    raw.data_icon = calloc(batch*3*32*32, sizeof(float));
    raw.data_truth = calloc(batch*10*1*1, sizeof(float));
    raw.label = calloc(batch, sizeof(int));
    return raw;
}

void free_batch_data(batch_data raw)
{
    if (raw.data_icon) free(raw.data_icon);
    if (raw.data_truth) free(raw.data_truth);
    if (raw.label) free(raw.label);
}

void append_cifa_data(batch_data* batch_cifa, cifa_data new_cifa)
{
    int batch = batch_cifa->batch;
    int c = 3;
    int h = 32;
    int w = 32;
    int out_c = 10;
    int out_h = 1;
    int out_w = 1;
    float* truth_value = get_cifa_output(new_cifa,out_c,out_h,out_w);

    memcpy(batch_cifa->data_icon + batch*c*h*w, new_cifa.icon.data, c*h*w*sizeof(float));
    memcpy(batch_cifa->data_truth + batch*out_c*out_h*out_w, truth_value , out_c*out_h*out_w*sizeof(float));
    batch_cifa->label[batch] = new_cifa.label;
    batch_cifa->batch = batch+1;
}

void save_weights_upto(network net, const char *filename, int cutoff)
{
    if (net.gpu_index >= 0){
        cuda_set_device(net.gpu_index);
    }
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if (!fp){
        file_error(filename);
        printf("Unable to save: %s\n", filename);
    }
    int major = 0;
    int minor = 1;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net.seen, sizeof(int), 1, fp);

    int i;
    for (i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        }
    }
    fclose(fp);
}

void load_weights_upto(network *net, const char *filename, int cutoff)
{
    if (net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
    fprintf(stderr, "Loading weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp){
        printf("Unable to load: %s\n", filename);
        return;
    }

    int major;
    int minor;
    int revision;
    size_t status = 1;
    status *= fread(&major, sizeof(int), 1, fp);
    status *= fread(&minor, sizeof(int), 1, fp);
    status *= fread(&revision, sizeof(int), 1, fp);
    status *= fread(net->seen, sizeof(int), 1, fp);
    if (status == 0) printf("Error with fread in load_weights_upto...\n");

    int i;
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void shuffle_list(const char* filename)
{
    FILE* fp;
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    char** list;
    int list_cnt;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    list = calloc(50000, sizeof(char*));
    list_cnt = 0;
    while ((read = getline(&line, &len, fp)) != -1){
        list[list_cnt] = calloc(256, sizeof(char));
        memcpy(list[list_cnt], line, (size_t)read*sizeof(char));
        list_cnt++;
    }
    fclose(fp);

    fp = fopen(filename, "w");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    list_cnt = 0;
    while (list_cnt < 50000){
        int rand_num = rand()%50000;
        if (list[rand_num] != NULL){
            fputs(list[rand_num], fp);
            list[rand_num] = NULL;
            list_cnt++;
        }
    }
    fclose(fp);
    if (line) free(line);
}

int main(int argc, char** argv)
{
    const char weight_filename[] = "/home/Documents/proto/proto.weights";
    const char train_list_filename[] = "/home/Documents/proto/train.list";
    const char test_list_filename[] = "/home/Documents/proto/test.list";

    FILE* fp;
    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    clock_t start_t, end_t, total_t;

    float cost;
    double total_train_error;

    int cnt = 0;

    network net = create_network();
    load_weights_upto(&net, weight_filename, 0);

    batch_data input = make_batch_data(net.batch);

    printf("****************\n");
    printf("TRAINING\n");
    printf("****************\n");
    start_t = clock();
    while (get_current_batch(net) < net.max_batches || net.max_batches == 0){
        total_train_error=0;
        fp = fopen(train_list_filename, "r");
        if (fp == NULL) exit(EXIT_FAILURE);

        while ((read = getline(&line, &len, fp)) != -1){
            //printf("Retrieved line of length %zu :\n", read);
            //printf("%s", line);
            cnt += 1;
            line[strlen(line)-1] = 0;
            cifa_data data = fetch_cifa_data(line);
            append_cifa_data(&input, data);
            if (cnt%net.batch==0){
                cost = train_network_datum_gpu(net, input.data_icon, input.data_truth);
                total_train_error += cost;
                printf("current_batch: %d, train_error: %f\n", get_current_batch(net), total_train_error);
                input.batch = 0;
            }
        }
        fclose(fp);
        end_t = clock();
        total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;

        save_weights_upto(net, weight_filename, 0);
        shuffle_list(train_list_filename);

        printf("last_train_error: %f\n", total_train_error);
        printf("Total time taken by CPU: %fs\n", (double)total_t);
        printf("\nFinished one epoch...\n\n");
    }

    printf("**************\n");
    printf("TESTING\n");
    printf("**************\n");
    double total_test_error;
    int total_match;

    line = NULL;
    input.batch = 0;
    total_test_error = 0;
    total_match = 0;
    cnt = 0;

    fp = fopen(test_list_filename, "r");
    if (fp == NULL) exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1){
        cnt += 1;
        line[strlen(line)-1] = 0;
        cifa_data data = fetch_cifa_data(line);
        append_cifa_data(&input, data);
        if (cnt%net.batch==0 || cnt==10000){

            cost = test_network_datum_gpu(net, input.data_icon, input.data_truth);
            total_test_error += cost;

            float* error = network_predict_gpu(net, input.data_icon);
            int num = 0, max_label = 0;
            float max = 0;
            for (num=0; num < net.batch*10; num++){
                if (error[num] > max){
                    max_label = num%10;
                    max = error[num];
                }
                if ((num+1)%10==0){
                    if (max_label==input.label[num/10]) total_match += 1;
                    max = 0;
                    max_label = 0;
                }
            }
            input.batch = 0;
        }
    }
    fclose(fp);
    end_t = clock();
    if (line) free(line);

    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("total_match: %d\n", total_match);
    printf("total_test_error: %f\n", total_test_error);
    printf("Total time taken by CPU: %fs\n", (double)total_t);

    printf("Exiting of the program...\n");
    return 0;
}
