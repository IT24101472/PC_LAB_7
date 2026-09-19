CC = mpicc
MPIRUN = OMPI_MCA_btl=self,sm mpirun
NP = 4

all:
	$(CC) -o Exercise01/sum_bcast Exercise01/sum_bcast.c
	$(CC) -o Exercise02/sum_scatter Exercise02/sum_scatter.c
	$(CC) -o Exercise03/sum_gather Exercise03/sum_gather.c
	$(CC) -o Exercise04/sum_reduce Exercise04/sum_reduce.c
	$(CC) -o Exercise05/sum_allreduce Exercise05/sum_allreduce.c
	$(CC) -o Exercise06/sum_scan Exercise06/sum_scan.c

run: all
	$(MPIRUN) -np $(NP) ./Exercise01/sum_bcast
	$(MPIRUN) -np $(NP) ./Exercise02/sum_scatter
	$(MPIRUN) -np $(NP) ./Exercise03/sum_gather
	$(MPIRUN) -np $(NP) ./Exercise04/sum_reduce
	$(MPIRUN) -np $(NP) ./Exercise05/sum_allreduce
	$(MPIRUN) -np $(NP) ./Exercise06/sum_scan

clean:
	rm -f Exercise01/sum_bcast
	rm -f Exercise02/sum_scatter
	rm -f Exercise03/sum_gather
	rm -f Exercise04/sum_reduce
	rm -f Exercise05/sum_allreduce
	rm -f Exercise06/sum_scan
