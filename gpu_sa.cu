
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
// ---------------------- 載入 QUBO 檔案 ----------------------
float* load_qubo(const char* path, int* n) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }
    
    // 讀取 n
    fscanf(fp, "%d", n);
    
    // 配置空間並讀取 Q
    float* Q = (float*)malloc((*n) * (*n) * sizeof(float));
    if (!Q) {
        fprintf(stderr, "Failed to allocate Q.\n");
        exit(1);
    }
    for (int i = 0; i < (*n) * (*n); ++i) {
        fscanf(fp, "%f", &Q[i]);
    }
    fclose(fp);
    return Q;
}

// ---------------------- 計算當前解的能量 ----------------------
// 在 GPU 上使用時，請保證此呼叫在單一執行緒內部或以共享記憶體加速。
__device__ float compute_energy(const float* Q, const int* spins, int n) {
    float e = 0.0f;
    // x^T Q x
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // spins[i]、spins[j] 為 0/1，計算時可轉成 ±1 或直接 0/1
            // 此處直接以 0/1 計算: energy = sum(Q[i,j]*spins[i]*spins[j])
            e += Q[i * n + j] * spins[i] * spins[j];
        }
    }
    return e;
}

// ---------------------- 翻轉某個位元後能量變化量 ----------------------
// 此處的 ΔE 計算方法 (若 Q 對稱可簡化)，以 O(n) 做範例。
__device__ float delta_energy(const float* Q, const int* spins, int n, int flip_index) {
    // flips[flip_index] 從 0->1 或 1->0
    // 假設舊值為 oldVal, 新值為 newVal = 1 - oldVal
    // ΔE = (newVal - oldVal)* (Σ_j Q[flip_index, j]*spins[j] + Σ_i spins[i]*Q[i, flip_index])
    // 如果 Q 對稱，可以只算一次
    // 若 spins[flip_index] = s，則 newVal = 1-s, (newVal - oldVal) = (1-2s)
    
    int oldVal = spins[flip_index];       // s
    int newVal = 1 - oldVal;     // 1-s
    int diff   = newVal - oldVal;  // (1-2s) 

    // 若要完整考慮 Q[i,j] 跟 Q[j,i] 都可能有值
    float sum_i = 0.0f;
    float sum_j = 0.0f; 

    // (A) 計算 ∑(j=0..n-1) Q[i,j]*x_j
    //     其中 i 固定在 row
    for (int j = 0; j < n; j++) {
        sum_i += Q[flip_index * n + j] * spins[j];
    } 

    // (B) 計算 ∑(j=0..n-1) Q[j,i]*x_j
    //     i 固定在 column
    //     如果你知道下三角為0，而 i<j 時 Q[j,i]=0，可省略這些運算
    for (int j = 0; j < n; j++) {
        sum_j += Q[j * n + flip_index] * spins[j];
    } 

    return diff * (sum_i + sum_j); 
    
}

// ---------------------- CUDA kernel: 同時做多條解的模擬退火 ----------------------
__global__ void sa_kernel(
    const float* d_Q,    // QUBO matrix
    int* d_spins,        // 各條解的自變量，以 row-major: 整體大小 num_solutions*n
    float* d_energies,   // 各條解的能量
    int   n,
    int   max_iter,
    float T0, 
    float alpha,
    int   num_solutions,
    curandState* states  // 亂數引擎
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_solutions) return;
    
    // 取得本執行緒對應的解 (spins) 起始位置
    int offset = tid * n;
    // 將初始能量算一次
    float energy = compute_energy(d_Q, &d_spins[offset], n);
    
    // 模擬退火迭代
    for (int iter = 0; iter < max_iter; iter++) {
        float T = 1.0f - (float)(iter + 1) / max_iter;

        // 隨機選擇要翻轉的位元 (0~n-1)
        int flip_index = (int)(curand_uniform(&states[tid]) * n);
        float energy_new = compute_energy(d_Q, &d_spins[offset], n);
        // 計算翻轉後能量差
        float dE = delta_energy(d_Q, &d_spins[offset], n, flip_index);
        // 若能量降低就接受，否則以 e^{-dE/T} 機率接受
        if (dE < 0.0f) {
            // 接受翻轉
            d_spins[offset + flip_index] = 1 - d_spins[offset + flip_index];
            energy += dE;
            iter--;
            continue;
        } else {
            float r = curand_uniform(&states[tid]);
            if (r < expf(-dE / T)) {
                // 以機率接受翻轉
                d_spins[offset + flip_index] = 1 - d_spins[offset + flip_index];
                energy += dE;
            }
        }
    }
    
    // 儲存最終能量
    d_energies[tid] = energy;
}

// ---------------------- 初始化 curand 狀態 ----------------------
__global__ void init_curand(curandState* states, unsigned long seed, int num_solutions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_solutions) {
        // 每個 thread 使用不同的 seed，常見做法是 seed + tid
        curand_init(seed + tid, 0, 0, &states[tid]);
    }
}

float compute_energy_cpu(const float* Q, const int* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum += Q[i * n + j] * x[i] * x[j];
        }
    }
    return sum;
}

// ---------------------- 主程式 ----------------------
int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("用法: %s <qubo_file_path> <num_solutions> <max_iterations>\n", argv[0]);
        return 0;
    }
    clock_t start, end;
    const char* qubo_path = argv[1];
    int num_solutions = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    
    // 1. 讀取 QUBO
    int n;
    float* Q = load_qubo(qubo_path, &n);
    // printf("成功讀取 QUBO matrix, 大小 = %d x %d\n", n, n);

    // 建立主機端 spins 與能量陣列
    int* h_spins = (int*)malloc(num_solutions * n * sizeof(int));
    float* h_energies = (float*)malloc(num_solutions * sizeof(float));

    // 隨機初始化主機端 spins (每條解)
    for (int sol = 0; sol < num_solutions; sol++) {
        for (int i = 0; i < n; i++) {
            h_spins[sol * n + i] = rand() % 2;  // 0 or 1
        }
    }

    // 2. 配置 GPU 記憶體
    float* d_Q;
    int* d_spins;
    float* d_energies;
    curandState* d_states;

    cudaMalloc((void**)&d_Q, n * n * sizeof(float));
    cudaMalloc((void**)&d_spins, num_solutions * n * sizeof(int));
    cudaMalloc((void**)&d_energies, num_solutions * sizeof(float));
    cudaMalloc((void**)&d_states, num_solutions * sizeof(curandState));

    // 複製 Q 與 spins 到 GPU
    cudaMemcpy(d_Q, Q, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spins, h_spins, num_solutions * n * sizeof(int), cudaMemcpyHostToDevice);

    // 3. 初始化 curand 狀態
    int blockSize = 256;
    int gridSize  = (num_solutions + blockSize - 1) / blockSize;
    init_curand<<<gridSize, blockSize>>>(d_states, 1234UL, num_solutions);
    cudaDeviceSynchronize();

    // 4. 設定模擬退火參數 (可依需求調整)
    float T0   = 10.0f;  // 初始溫度
    float alpha= 0.99f;  // 降溫係數
    start = clock();
    // 5. 在 GPU 上進行模擬退火
    sa_kernel<<<gridSize, blockSize>>>(
        d_Q, d_spins, d_energies,
        n, max_iter, T0, alpha, num_solutions,
        d_states
    );
    cudaDeviceSynchronize();
    end = clock();

    // 6. 將結果帶回主機端
    cudaMemcpy(h_energies, d_energies, num_solutions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spins, d_spins, num_solutions * n * sizeof(int), cudaMemcpyDeviceToHost);

    // 在主機端找出最佳解 (最低能量)
    float bestE = h_energies[0];
    int bestSol = 0;
    for (int sol = 1; sol < num_solutions; sol++) {
        if (h_energies[sol] < bestE) {
            bestE = h_energies[sol];
            bestSol = sol;
        }
    }

    // 輸出結果
    printf("Energy: %f, Time: %f\n", compute_energy_cpu(Q,h_spins + (bestSol * n), n), ((double)(end - start)) / CLOCKS_PER_SEC);
    /*
    printf("對應解 (spins): ");
    for (int i = 0; i < n; i++) {
        printf("%d", h_spins[bestSol * n + i]);
        if (i < n - 1) printf(", ");
    }
    printf("\n");
    */
    // 釋放記憶體
    cudaFree(d_Q);
    cudaFree(d_spins);
    cudaFree(d_energies);
    cudaFree(d_states);
    free(Q);
    free(h_spins);
    free(h_energies);

    return 0;
}

