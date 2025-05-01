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


