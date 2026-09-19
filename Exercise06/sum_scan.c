#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * Make sure N can be divided evenly
     * between all MPI processes.
     */
    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: Number of processes must evenly divide %d\n", N);
        }

        MPI_Finalize();
        return 1;
    }

    int chunk_size = N / size;

    /*
     * Only Rank 0 stores the full array.
     */
    int *array = NULL;

    if (rank == 0) {

        array = (int *)malloc(N * sizeof(int));

        for (int i = 0; i < N; i++) {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    /*
     * Every process stores only its own chunk.
     */
    int *local_chunk =
        (int *)malloc(chunk_size * sizeof(int));

    /*
     * Start timing before communication.
     */
    double start = MPI_Wtime();

    /*
     * Divide the original array among all processes.
     */
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

    /*
     * Calculate local sum.
     */
    long long local_sum = 0;

    for (int i = 0; i < chunk_size; i++) {
        local_sum += local_chunk[i];
    }

    /*
     * MPI_Scan performs a prefix reduction.
     *
     * Rank 0 gets:
     * local_sum0
     *
     * Rank 1 gets:
     * local_sum0 + local_sum1
     *
     * Rank 2 gets:
     * local_sum0 + local_sum1 + local_sum2
     *
     * etc.
     */
    long long prefix_sum = 0;

    MPI_Scan(
        &local_sum,
        &prefix_sum,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    /*
     * Total contribution from all ranks
     * before the current rank.
     */
    long long sum_before_me =
        prefix_sum - local_sum;

    printf(
        "Rank %d: local_sum = %lld, prefix_sum = %lld, sum_before_me = %lld\n",
        rank,
        local_sum,
        prefix_sum,
        sum_before_me
    );

    /*
     * The last rank's prefix_sum is the
     * total sum of the whole array.
     */
    if (rank == size - 1) {

        double elapsed = MPI_Wtime() - start;

        long long expected =
            (long long)N * (N + 1) / 2;

        printf("\n[Scan] Final prefix sum = %lld\n", prefix_sum);
        printf("[Scan] Expected         = %lld\n", expected);

        printf(
            "[Scan] Correct?          = %s\n",
            prefix_sum == expected ? "YES" : "NO"
        );

        printf("[Scan] Time              = %.4f sec\n", elapsed);
    }

    /*
     * Free allocated memory.
     */
    if (rank == 0) {
        free(array);
    }

    free(local_chunk);

    MPI_Finalize();

    return 0;
}
