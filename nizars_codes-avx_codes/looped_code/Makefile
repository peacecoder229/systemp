CC=g++ -std=c++11 -mavx2 -fopenmp -g
FLAGs=-std=c++11 -fopenmp -g

all: clean 04-dot-product cpu_stress_avx cpu_stress_no_avx cpu_stress_avx512

04-dot-product: 04-dot-product.c
	$(CC) $(FLAGs) 04-dot-product.c -o 04-dot-product

cpu_stress_avx: cpu_stress_avx.cpp
	$(CC) $(FLAGs) cpu_stress_avx.cpp -o cpu_stress_avx

cpu_stress_avx512: cpu_stress_avx512.cpp
	$(CC) $(FLAGs) -mavx512f -mavx512er -mavx512pf -mavx512cd cpu_stress_avx512.cpp -o 512_cpu_stress_avx 

cpu_stress_no_avx: cpu_stress_no_avx.cpp
	$(CC) $(FLAGs) cpu_stress_no_avx.cpp -o cpu_stress_no_avx

clean:
	rm -f 04-dot-product cpu_stress_avx cpu_stress_no_avx 512_cpu_stress_avx

