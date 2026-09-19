#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    // Only root has the complete array
    int *array = NULL;

    if (rank == 0) {

        array = (int *)malloc(N * sizeof(int));

        for (int i = 0; i < N; i++) {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    // Every process receives only its own chunk
    int *local_chunk =
        (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    // Distribute chunks
    MPI_Scatter(
        array,
        chunk_size,
        MPI_INT,
        local_chunk,
        chunk_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Each process calculates its own sum
    long long local_sum = 0;

    for (int i = 0; i < chunk_size; i++) {
        local_sum += local_chunk[i];
    }

    printf(
        "Rank %d: local_sum = %lld\n",
        rank,
        local_sum
    );

    /*
     * Only root needs this array.
     * It stores one local_sum from each process.
     */
    long long *all_sums = NULL;

    if (rank == 0) {
        all_sums =
            (long long *)malloc(size * sizeof(long long));
    }

    /*
     * Collect local_sum from every process.
     */
    MPI_Gather(
        &local_sum,
        1,
        MPI_LONG_LONG,
        all_sums,
        1,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Only root receives all_sums.
     */
    if (rank == 0) {

        long long total_sum = 0;

        for (int i = 0; i < size; i++) {
            total_sum += all_sums[i];
        }

        double elapsed = MPI_Wtime() - start;

        long long expected =
            (long long)N * (N + 1) / 2;

        printf("\n[Gather] Total sum = %lld\n", total_sum);
        printf("[Gather] Expected  = %lld\n", expected);
        printf(
            "[Gather] Correct?   = %s\n",
            total_sum == expected ? "YES" : "NO"
        );
        printf("[Gather] Time      = %.4f sec\n", elapsed);

        free(all_sums);
    }

    if (rank == 0) {
        free(array);
    }

    free(local_chunk);

    MPI_Finalize();

    return 0;
}
