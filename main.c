#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>


double get_num(double b, double a) {
    double delta = fabs(a - b);
    double min = a;
    if (a > b)
        min = b;
    return ((double) rand() * delta / RAND_MAX + min);
}

typedef struct matrix {
    bool init;
    int height;
    int width;
    double **data;
} matrix;

matrix *init_matrix(int height, int width, double init_value) {
    matrix *m = (matrix *) malloc(sizeof(matrix));
    m->init = false;
    m->height = height;
    m->width = width;
    m->data = (double **) calloc(height, sizeof(double *));
    for (int i = 0; i < height; i++) {
        m->data[i] = (double *) calloc(width, sizeof(double));
        for (int j = 0; j < m->width; j++) {
            m->data[i][j] = init_value;
        }
    }
    m->init = true;
    return m;
}

void print_matrix(const matrix *m) {
    printf("_+_+_+_+_+_+_+_\n");
    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            printf("|%f", m->data[i][j]);
        }
        printf("|\n");
    }
    printf("_+_+_+_+_+_+_+_\n");
}

matrix *multiply(const matrix *m1, const matrix *m2) {
    if (m1->width != m2->height || m1->init == false || m2->init == false)
        return 0;
    matrix *ret = init_matrix(m1->height, m2->width, 0);
    for (int i = 0; i < ret->height; i++) {
        for (int j = 0; j < ret->width; j++) {
            for (int k = 0; k < m2->height; k++) {
                ret->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
    return ret;
}


matrix *addition(const matrix *m1, const matrix *m2) {
    if (m1->height != m2->height || m1->width != m2->width || m1->init == false || m2->init == false)
        return 0;
    int sizes[2];
    sizes[0] = m1->height;
    sizes[1] = m1->width;
    matrix *ret = init_matrix(m1->height, m1->width, 0);

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            ret->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }

    return ret;
}

double matrix_sum(const matrix *m1) {
    double ret = 0;
    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            ret += m1->data[i][j];
        }
    }
    return ret;
}

matrix *subtraction(const matrix *m1, const matrix *m2) {
    if (m1->height != m2->height || m1->width != m2->width || m1->init == false || m2->init == false)
        return 0;
    int sizes[2];
    sizes[0] = m1->height;
    sizes[1] = m1->width;
    matrix *ret = init_matrix(m1->height, m1->width, 0);

    for (int i = 0; i < m1->height; i++) {
        for (int j = 0; j < m1->width; j++) {
            ret->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }

    return ret;
}

bool set_data(matrix *m, double **data) {
    if (m->init == false)
        return false;

    for (int i = 0; i < m->height; i++) {
        for (int j = 0; j < m->width; j++) {
            m->data[i][j] = data[i][j];
        }
    }
    return true;
}


matrix *copy_matrix(matrix *m1) {
    matrix *ret = init_matrix(m1->height, m1->width, 0);
    set_data(ret, m1->data);
    return ret;
}

matrix *multiply_matrix_vals(matrix *m1, double scalar) {
    matrix *ret = copy_matrix(m1);
    for (int i = 0; i < ret->height; i++)
        for (int j = 0; j < ret->width; j++)
            ret->data[i][j] *= scalar;
    return ret;
}

matrix *matrix_transpose(matrix *m1) {
    matrix *ret = init_matrix(m1->width, m1->height, 0);
    for (int i = 0; i < ret->height; i++)
        for (int j = 0; j < ret->width; j++)
            ret->data[i][j] = m1->data[j][i];
    return ret;
}

matrix *matrix_pow(matrix *m1, double power) {
    matrix *ret = copy_matrix(m1);
    for (int i = 0; i < ret->height; i++)
        for (int j = 0; j < ret->width; j++)
            ret->data[i][j] = pow(ret->data[i][j], power);
    return ret;
}

matrix *matrix_abs(matrix *m1) {
    matrix *ret = copy_matrix(m1);
    for (int i = 0; i < ret->height; i++)
        for (int j = 0; j < ret->width; j++)
            ret->data[i][j] = fabs(ret->data[i][j]);
    return ret;
}

typedef struct layer {
    matrix *weights;

    double (*activation)(double);

    double (*activation_derivative)(double);

    double *biases;
    bool init;
    int input_size;
    int output_size;
    bool use_bias;
} layer;


double linear(double x)
{
    return x;
}

double linear_div(double x){
    return 1;
}
double relu(double x) {
    if (x < 0.0)
        return 0;
    return x;
}

double relu_derivative(double x) {
    if (x < 0.0)
        return 0;
    return 1;
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

layer *
init_layer(double (*activation)(double), double (*activation_derivative)(double), int input_size, int output_size,bool use_bias) {
    layer *l = (layer *) malloc(sizeof(layer));
    l->activation = activation;
    l->activation_derivative = activation_derivative;
    l->biases = calloc(output_size, sizeof(double));
    l->init = true;
    l->use_bias = use_bias;
    l->output_size = output_size;
    l->input_size = input_size;
    l->weights = init_matrix(output_size, input_size, 0);
    srand(time(NULL));
    double lim = pow(6.0 / input_size + output_size, 0.5);
    for (int i = 0; i < output_size; i++) {
        l->biases[i] = 0;
        for (int j = 0; j < input_size; j++) {
            l->weights->data[i][j] = get_num(-lim, lim);
        }
    }
    return l;
}

void print_layer(const layer *l1) {
    printf("This layer has %d inputs, %d outputs\nWeights:", l1->input_size, l1->output_size);
    print_matrix(l1->weights);
    printf("biases:\n");
    for(int i = 0 ; i < l1->output_size;i++){
        printf("%f\t",l1->biases[i]);
    }

}

matrix *weighted_sum(const layer *l1, matrix *inputs) {
    matrix *ret = multiply(l1->weights, inputs);
    return ret;
}

matrix *activated_sum(const layer *l1, matrix *inputs) {
    matrix *weighted = weighted_sum(l1, inputs);
    for (int i = 0; i < weighted->height; i++) {
        weighted->data[i][0] += l1->biases[i];
//        weighted->data[i][0] = l1->activation(weighted->data[i][0]);
    }

    return weighted;
}

typedef struct network {
    layer **layers;
    int network_size;
    double learning_rate;
    double weight_decay;

    matrix *(*error)(matrix *, matrix *);

    matrix *(*error_derivative)(matrix *, matrix *);

    bool init;
} network;

void print_network(network *net) {
    for (int i = 0; i < net->network_size - 1; i++) {
        print_layer(net->layers[i]);
    }
}

network *init_network(const int *layer_sizes, int network_size, double (**activations)(double),
                      double (**activation_derivatives)(double), matrix *(*error)(matrix *, matrix *),
                      matrix *(*error_derivative)(matrix *, matrix *), double learning_rate, double weight_decay,bool use_bias) {
    network *net = (network *) malloc(sizeof(network));
    net->network_size = network_size;
    net->error = error;
    net->error_derivative = error_derivative;
    net->init = true;
    net->layers = (layer **) calloc(network_size - 1, sizeof(layer *));
    for (int i = 0; i < network_size - 1; i++) {
        net->layers[i] = init_layer(activations[i], activation_derivatives[i], layer_sizes[i], layer_sizes[i + 1],use_bias);
    }
    net->learning_rate = learning_rate;
    net->weight_decay = weight_decay;
    return net;
}


matrix *feed_forward(layer *l, matrix *inputs) {
    matrix *ret = activated_sum(l, inputs);
    for (int i = 0; i < ret->height; i++)
        for (int j = 0; j < ret->height; j++)
            ret->data[i][j] = l->activation(ret->data[i][j]);
    return ret;
}


matrix *mse_derivative(matrix *prediction, matrix *true_value) {
    matrix *ret = subtraction(true_value, prediction);
    return ret;
}


double mae_derivative(matrix *prediction, matrix *true_value) {
    if (prediction > true_value)
        return 1;
    return -1;
}


matrix *mse(matrix *prediction, matrix *true_value) {
    matrix *temp = subtraction(prediction, true_value);
    matrix *ret = matrix_pow(temp, 2);
    free(temp);
    temp = ret;
    ret = multiply_matrix_vals(ret, 1.0 / prediction->height);
    free(temp);
    return ret;
}

matrix *mae(matrix *prediction, matrix *true_value) {
    matrix *temp = subtraction(prediction, true_value);
    matrix *ret = matrix_abs(temp);
    free(temp);
    return ret;
}


matrix *get_losses(layer *l, matrix *errors) {
    matrix *weight_sums = init_matrix(l->output_size, 1, 0);
    for (int i = 0; i < l->input_size; i++) {
        for (int j = 0; j < l->output_size; j++)
            weight_sums->data[j][0] += l->weights->data[j][i];
    }
    matrix *fractioned_weights = copy_matrix(l->weights);
    for (int i = 0; i < fractioned_weights->height; i++) {
        for (int j = 0; j < fractioned_weights->width; j++)
            fractioned_weights->data[i][j] /= weight_sums->data[i][0];
    }
    matrix *weighted_errors = multiply(errors, fractioned_weights);
    print_matrix(errors);
    print_matrix(weight_sums);
    print_matrix(fractioned_weights);
    print_matrix(weighted_errors);
    free(weight_sums);
    free(fractioned_weights);
    return weighted_errors;
}


void backpropogate(network *net, matrix *inputs, matrix *true_value) {
    matrix *temp = inputs;
    matrix **outputs = calloc(net->network_size, sizeof(matrix *));
    outputs[0] = copy_matrix(temp);
    matrix **nets = calloc(net->network_size - 1, sizeof(matrix *));
    matrix *prediction;
    // getting the predictions and nets
    for (int i = 0; i < net->network_size - 1; i++) {
        nets[i] = activated_sum(net->layers[i], temp);
        prediction = copy_matrix(nets[i]);
        for (int j = 0; j < prediction->height; j++)
            for (int k = 0; k < prediction->width; k++)
                prediction->data[j][k] = net->layers[i]->activation(prediction->data[j][k]);
        outputs[i + 1] = copy_matrix(prediction);
        free(prediction);
        temp = outputs[i + 1];
    }
    //printing some info
    printf("\n\n\nloss:%f\ninputs:",net->error(temp,true_value)->data[0][0]);
    print_matrix(inputs);
    printf("outputs:");
    print_matrix(temp);
    printf("answer:");
    print_matrix(true_value);
//    print_network(net);
    matrix *losses = net->error_derivative(temp, true_value);
    prediction = nets[net->network_size - 2];
    matrix *deltas_l = init_matrix(net->layers[net->network_size - 2]->output_size, 1, 0);
    matrix *gradiants = init_matrix(net->layers[net->network_size - 2]->output_size,
                                    net->layers[net->network_size - 2]->input_size, 0);
    double * bias_grads = calloc(net->layers[net->network_size - 2]->output_size,sizeof(double));
    double sum = matrix_sum(net->layers[net->network_size - 2]->weights);
    // the output layer
    for (int i = 0; i < net->layers[net->network_size - 2]->output_size; i++) {
        deltas_l->data[i][0] =
                losses->data[i][0] * net->layers[net->network_size - 2]->activation_derivative(prediction->data[i][0]);
        bias_grads[i] = deltas_l->data[i][0] * net->learning_rate;
        for (int j = 0; j < net->layers[net->network_size - 2]->input_size; j++) {
//            net->layers[net->network_size-2]->weights->data[i][j] -= output * deltas_l->data[i][0];
            gradiants->data[i][j] =
                    outputs[net->network_size - 2]->data[j][0] * deltas_l->data[i][0];
        }
    }
    // the non output layers
    matrix *deltas_j;
    for (int i = net->network_size - 3; i >= 0; i--) {
        sum = matrix_sum(net->layers[i]->weights);
        deltas_j = init_matrix(net->layers[i]->output_size, 1, 0);
        // calculating the lambdas for the j layer
        for (int j = 0; j < net->layers[i]->output_size; j++) {
            for (int k = 0; k < deltas_l->height; k++) {
                //getting the lambdas
                deltas_j->data[j][0] += net->layers[i + 1]->weights->data[k][j] * deltas_l->data[k][0];
                //applying the gradiants
//                net->layers[i+1]->weights->data[k][j] -= gradiants->data[k][j] *net->learning_rate;
            }
            deltas_j->data[j][0] *= net->layers[i]->activation_derivative(nets[i]->data[j][0]);
        }
//        print_matrix(deltas_j);
        //applying the prev gradiants
//        print_matrix(gradiants);
        temp = gradiants;
        gradiants = multiply_matrix_vals(temp, net->learning_rate);
        free(temp);
        temp = net->layers[i + 1]->weights;
        net->layers[i + 1]->weights = addition(temp, gradiants);
        free(temp);
        free(gradiants);
        //applying bias grads
        for (int j = 0; j < net->layers[i + 1]->output_size; ++j) {
            net->layers[i + 1]->biases[j] -= bias_grads[j]*net->layers[i]->use_bias;
        }
        free(bias_grads);
        // getting the gradiants
        bias_grads = calloc(net->layers[i]->output_size,sizeof(double));
        gradiants = init_matrix(net->layers[i]->output_size, net->layers[i]->input_size, 0);
        for (int j = 0; j < net->layers[i]->output_size; j++) {
            bias_grads[j] = deltas_j->data[j][0] * net->learning_rate;
            for (int k = 0; k < net->layers[i]->input_size; k++) {
                gradiants->data[j][k] = outputs[i]->data[k][0] * deltas_j->data[j][0];
            }
        }
        deltas_l = deltas_j;
    }
    // applying the last gradiants
    temp = gradiants;
    gradiants = multiply_matrix_vals(temp, net->learning_rate);
    free(temp);
    temp = net->layers[0]->weights;
    net->layers[0]->weights = addition(temp, gradiants);
    //applying the last bias gradiants
    for (int j = 0; j < net->layers[0]->output_size; ++j) {
        net->layers[0]->biases[j] -= bias_grads[j]*net->layers[0]->use_bias;
    }
    free(bias_grads);

    free(gradiants);
    free(deltas_l);
    free(temp);
    free(outputs);
    free(nets);
    free(losses);
    free(prediction);
}


int main() {
    int net_size = 4;
    int *layer_sizes = calloc(net_size,sizeof(int));
    layer_sizes[0] = 1;
    layer_sizes[1] = 8;
    layer_sizes[2] = 16;
    layer_sizes[3] = 1;
    double (**activations)(double) = calloc(net_size-1,sizeof(double(*)(double)));
    activations[0] = sigmoid;
    activations[1] = sigmoid;
    activations[2] = linear;
    double (**activations_div)(double) = calloc(net_size-1,sizeof(double(*)(double)));
    activations_div[0] = sigmoid_derivative;
    activations_div[1] = sigmoid_derivative;
    activations_div[2] = linear_div;
    network *net = init_network(layer_sizes, net_size, activations, activations_div, mse, mse_derivative, 0.001, 0.01,false);
    int data_size = 500000;
    matrix **inputs = calloc(data_size, sizeof(matrix *));
    matrix **true_vals = calloc(data_size, sizeof(matrix *));
    double num = 0;
    for (int i = 0; i < data_size; i++) {
        num = get_num(0.0, 10.0);
        inputs[i] = init_matrix(1, 1, num);
        true_vals[i] = init_matrix(1, 1, num *2);
//        if (inputs[i]->data[0][0] >= 3.0){
//            true_vals[i] = init_matrix(1,1, 0);
//        }
//        else{
//            true_vals[i] = init_matrix(1,1, 10);
//        }
//        print_network(net);
    }
    for (int i = 0; i < data_size; i++) {
        backpropogate(net, inputs[i], true_vals[i]);
    }
    free(inputs);
    free(true_vals);
    free(net);
    return 0;
}
