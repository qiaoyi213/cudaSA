#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
typedef struct {
    size_t row;
    size_t col;
    float* elements;
} matrix_t;

matrix_t matrix_create(size_t row, size_t col) {
    matrix_t matrix = {0};
    matrix.row = row;
    matrix.col = col;
    matrix.elements = (float*)malloc(row * col * sizeof(float));
    
    if (matrix.elements == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize all elements to 0
    memset(matrix.elements, 0, row * col * sizeof(float));
    return matrix;
}

matrix_t matrix_init(size_t row, size_t col, float* elements) {
    matrix_t matrix = matrix_create(row, col);
    
    if (elements != NULL) {
        memcpy(matrix.elements, elements, row * col * sizeof(float));
    }
    
    return matrix;
}

bool matrix_is_valid(const matrix_t matrix) {
    return matrix.elements != NULL && matrix.row > 0 && matrix.col > 0;
}

void matrix_print(const matrix_t matrix) {
    if (!matrix_is_valid(matrix)) {
        printf("Invalid matrix\n");
        return;
    }

    for (size_t i = 0; i < matrix.row; i++) {
        for (size_t j = 0; j < matrix.col; j++) {
            printf("%8.2f ", matrix.elements[i * matrix.col + j]);
        }
        printf("\n");
    }
}


float matrix_at(const matrix_t matrix, size_t i, size_t j) {
    return matrix.elements[i*matrix.col + j];
}

matrix_t matrix_set(const matrix_t matrix, size_t i, size_t j, double value) {
    matrix.elements[i*matrix.col + j] = value;
    return matrix;
}

// (M * K) * (K * N) = M * N
__global__ void cuda_matrix_mul(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

matrix_t matrix_mul(matrix_t A, matrix_t B) {
    // return the rusult of a * b
    float* d_a, * d_b, * d_c;
    int M = A.row, N = B.col, K = A.col;

    // Allocate device memory
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_a, A.elements, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B.elements, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    cuda_matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N, K);

    // Copy the result back to the host
    float* c = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    matrix_t C = matrix_init(M, N, c);
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(c);
    return C;
}

matrix_t matrix_transpose(const matrix_t A) {
    matrix_t result = matrix_create(A.col, A.row);
    
    for (size_t i = 0; i < A.row; i++) {
        for (size_t j = 0; j < A.col; j++) {
            result.elements[j * A.row + i] = A.elements[i * A.col + j];
        }
    }
    
    return result;
}

void matrix_free(matrix_t* matrix) {
    if (matrix != NULL && matrix->elements != NULL) {
        free(matrix->elements);
        matrix->elements = NULL;
        matrix->row = 0;
        matrix->col = 0;
    }
}
float calculate_hamiltonain(matrix_t Q, matrix_t x){
    matrix_t x_t = matrix_transpose(x);
    matrix_t m = matrix_mul(x_t, Q);
    matrix_t result = matrix_mul(m, x);
    return result.elements[0];
}

matrix_t neighbour(matrix_t x) {
    int len = x.row;
    int pos = rand() % len;
    x = matrix_set(x, 0, pos, -1 * (matrix_at(x, 0, pos) - 1));
    return x;
}

float temperature(int k, int k_max){
     
    return 1-((k+1)/k_max);
}

float P(double e, double en, double T) {
    if(en > e) return 1;
    return exp((e - en)/T);
}

matrix_t sa_sample(matrix_t Q) {
    matrix_t s = matrix_create(Q.row, 1);
    int k_max = 100;
    matrix_t sn;
    for(int k=0;k<k_max;k++) {
        double T = temperature(k, k_max);
        sn = neighbour(s);
        double e = calculate_hamiltonain(Q, s);
        double en = calculate_hamiltonain(Q, sn);
        if(rand() % 100 < P(e, en, T) * 100){
            s = sn;
            e = en;
        }
    }
    printf("Hamiltonain: %f \n", calculate_hamiltonain(Q, s));
    return s;
}

matrix_t load_qubo_txt(char* path) {
    
    FILE *fp = NULL;
    fp = fopen(path, "r");
    if(!fp) {
        printf("Load QUBO Error");
        exit(1);
    }
    int size = 0;
    fscanf(fp, "%d", &size);
    float* data = (float*)malloc((size*size)*sizeof(float));

    for(int i=0;i<size*size;i++){
        float num = 0.0;
        fscanf(fp, "%f", &num);
        data[i] = num;
    } 
    fclose(fp);
    return matrix_init(size, size, data);
}

int main(void) {
    matrix_t Q = load_qubo_txt("qubo.txt");
    matrix_print(Q);
    for(int i=0;i<10;i++){
        matrix_t x = sa_sample(Q);
        matrix_print(x);
    }
    return 0;
}
