#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define K_MAX 1000
#define MAX_PATH_LEN 1999000
float* load_qubo(const char* path, int* n_out){
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s\n", path);
        exit(EXIT_FAILURE);
    }

    int n;
    if (fscanf(fp, "%d", &n) != 1) {
        fprintf(stderr, "Failed to read dimension\n");
        exit(EXIT_FAILURE);
    }

    float* Q = (float*)malloc(n * n * sizeof(float));
    if (!Q) {
        fprintf(stderr, "Failed to allocate Q matrix\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n * n; ++i) {
        if (fscanf(fp, "%f", &Q[i]) != 1) {
            fprintf(stderr, "Error reading Q[%d]\n", i);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fp);
    *n_out = n;
    return Q;
}

float compute_energy(const float* Q, const int* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum += Q[i * n + j] * x[i] * x[j];
        }
    }
    return sum;
}

void flip_random_bit(int* x, int n) {
    int pos = rand() % n;
    x[pos] = 1 - x[pos];
}

float delta_energy(const float* Q, const int* spins, int n, int flip_index) {
    int oldVal = spins[flip_index];
    int newVal = 1 - oldVal;
    int diff = newVal - oldVal;

    float sum_i = 0.0f;
    float sum_j = 0.0f;

    for (int j = 0; j < n; j++) {
        sum_i += Q[flip_index * n + j] * spins[j];
    }

    for (int j = 0; j < n; j++) {
        sum_j += Q[j * n + flip_index] * spins[j];
    }

    return diff * (sum_i + sum_j);
}

int* sa_cpu(const float* Q, int n, float* best_energy_out) {
    int* x = (int*)malloc(n * sizeof(int));
    int* x_new = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        x[i] = rand() % 2;
        x_new[i] = x[i];
    }

    float energy = compute_energy(Q, x, n);

    for (int k = 0; k < K_MAX; ++k) {
        float T = 1.0f - (float)(k + 1) / K_MAX;

        memcpy(x_new, x, n * sizeof(int));
        
        int pos = rand() % n;
        x_new[pos] = 1 - x_new[pos];
        float dE = delta_energy(Q, x, n, pos);

        float accept_prob = expf(dE / (fmaxf(T, 1e-6f)));
        if(dE < 0) {
            memcpy(x, x_new, n * sizeof(int));
            energy += dE;
            k--;
            continue;
        } else {
            float r = ((float)rand() / RAND_MAX);
            if (r < accept_prob) {
                memcpy(x, x_new, n * sizeof(int));
                energy += dE;
            }
        }
    }

    free(x_new);
    *best_energy_out = energy;
    return x;
}

int parse_mcp(char* path, int* x, int n) {
    char data_path[MAX_PATH_LEN];
    strncpy(data_path, path, MAX_PATH_LEN);
    data_path[MAX_PATH_LEN-1] = '\0'; // 保證結束符

    // 替換 qubo 為 data
    char* p = strstr(data_path, "qubo");
    if (p != NULL) {
        memmove(p + 4, p + 4, strlen(p + 4) + 1); // 無實際意義, 只是保證有終止字元
        memcpy(p, "data", 4);
    } else {
        // 沒找到直接返回
        fprintf(stderr, "Cannot find 'qubo' in path!\n");
        return -1.0f;
    }

    FILE* fp = fopen(data_path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", data_path);
        return -1.0f;
    }

    int vertex_num;
    if (fscanf(fp, "%d", &vertex_num) != 1) {
        fprintf(stderr, "Failed to read vertex number from file.\n");
        fclose(fp);
        return -1.0f;
    }
    // Optionally check: vertex_num == n

    int sum = 0;
    int u, v;
    int weight;
    int E;
    fscanf(fp, "%d",&E);
    while (fscanf(fp, "%d %d %d", &u, &v, &weight) == 3) {
        if (u < 0 || v < 0 || u >= n || v >= n) continue; // skip out of range
        if(x[u-1] != x[v-1]) {
            sum += weight;
        }
    }
    fclose(fp);
    return sum;
}
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s qubo.txt [iterations]\n", argv[0]);
        return 1;
    }

    srand((unsigned int)time(NULL));
    clock_t start, end;
    int n;
    float* Q = load_qubo(argv[1], &n);
    int iterations = 400;
    if (argc > 2) {
        iterations = atoi(argv[2]);
    }
    float result = 100000;
    start = clock();
    int* best_sol = (int*)malloc(n * sizeof(int));
    for(int i=0;i<iterations;i++) {
        float best_energy;
        int* x = sa_cpu(Q, n, &best_energy);
        if(best_energy < result){
            result = best_energy;
            memcpy(best_sol, x, n * sizeof(int));
        }
        free(x);
    }
    end = clock();
    result = compute_energy(Q, best_sol, n);
    printf("Energy: %f, Time: %lf\n", result, ((double)(end - start)) / CLOCKS_PER_SEC);
    /*
    for(int i=0;i<n;i++){
        printf("%d ", best_sol[i]);
    }
    printf("\n");
    */
    printf("%d \n", parse_mcp(argv[1], best_sol,n));
    free(Q);
    free(best_sol);
    return 0;
}
