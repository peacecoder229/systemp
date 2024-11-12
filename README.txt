Running benchdnn  benchmark inside  test directory in benchdnn..
numactl -C 0-23 env OMP_NUM_THREADS=24 ./benchdnn --matmul --dt=bf16:bf16:bf16 --stag=ab --wtag=ab --dtag=ab 5600x50:50x50
abc indicates 3D matrix..
ad indicates 2D matrix..

 numactl -C 0-23 env OMP_NUM_THREADS=24 ./benchdnn --matmul --dt=bf16:bf16:bf16 --stag=abc --wtag=abc --dtag=abc 5120x50x50:5120x50x50


Added MPI use case  mpi_usecase.py
time mpiexec --allow-run-as-root -n 10 python3 mpi_usecase.py 100000000
Running matmul program
numactl -C 0-15 ./matmul_args --src_row 1619 --src_col 809 --weight_row 809 --weight_col 1619 --batch_size 1 --ompthreads 4

 ONEDNN_VERBOSE=1 numactl -C 0-31 ./matmul_sereilbf16 --src_row 16190 --src_col 16190 --weight_row 16190 --weight_col 16190 --batch_size 1 --ompthreads 32 --layer 3 --cachedata 0 & ONEDNN_VERBOSE=1 numactl -C 64-95 ./matmul_sereilbf16 --src_row 16190 --src_col 16190 --weight_row 16190 --weight_col 16190 --batch_size 1 --ompthreads 32 --layer 3 --cachedata 0
