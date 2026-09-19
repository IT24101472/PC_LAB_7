#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Every process creates space for the full array
    int *array = (int *)malloc(N * sizeof(int));

    // Only Rank 0 fills the array
    if (rank == 0) {

        for (int i = 0; i < N; i++) {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    // Send the full array from Rank 0 to every process
    MPI_Bcast(
        array,
        N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Calculate how much work each process gets
    int chunk_size = N / size;

    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;

    // Each process calculates its own partial sum
    long long local_sum = 0;

    for (int i = start_idx; i < end_idx; i++) {
        local_sum += array[i];
    }

    printf(
        "Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
        rank,
        start_idx,
        end_idx,
        local_sum
    );

    // Worker ranks send their local sums to Rank 0
    if (rank != 0) {

        MPI_Send(
            &local_sum,
            1,
            MPI_LONG_LONG,
            0,
            0,
            MPI_COMM_WORLD
        );

    } else {

        // Rank 0 starts with its own local sum
        long long total_sum = local_sum;

        // Rank 0 receives partial sums from all other ranks
        for (int r = 1; r < size; r++) {

            long long recv_sum;

            MPI_Recv(
                &recv_sum,
                1,
                MPI_LONG_LONG,
                r,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            total_sum += recv_sum;
        }

        double elapsed = MPI_Wtime() - start;

        long long expected =
            (long long)N * (N + 1) / 2;

        printf("\n[Bcast] Total sum = %lld\n", total_sum);
        printf("[Bcast] Expected  = %lld\n", expected);

        printf(
            "[Bcast] Correct?   = %s\n",
            total_sum == expected ? "YES" : "NO"
        );

        printf("[Bcast] Time      = %.4f sec\n", elapsed);
    }

    free(array);

    MPI_Finalize();

    return 0;
}
