CC=g++ -std=c++11 -mavx2 -fopenmp -g
ICBC=/opt/intel/compilers_and_libraries_2018.3.222/linux/bin/intel64/icpc
FLAGs=-std=c++11 -fopenmp -g 
all: clean 04-dot-product cpu_stress_avx cpu_stress_no_avx cpu_stress_avx512

04-dot-product: 04-dot-product.c
	$(ICBC) $(FLAGs) 04-dot-product.c -o 04-dot-product

cpu_stress_avx: cpu_stress_avx.cpp
	$(ICBC) $(FLAGs) cpu_stress_avx.cpp -o cpu_stress_avx -axAVX2

cpu_stress_avx512: cpu_stress_avx512.cpp
	#$(CC) cpu_stress_avx512.cpp -mprefer-vector-width=512 -o 512_cpu_stress_avx
	$(ICBC) -xCOMMON-AVX512 cpu_stress_avx512.cpp -o 512_cpu_stress_avx -fopenmp -g 
	#clang++ cpu_stress_avx512.cpp -o 512_cpu_stress_avx -mavx512f -mavx512er -mavx512pf -mavx512cd -g
cpu_stress_no_avx: cpu_stress_no_avx.cpp 
	$(ICBC) $(FLAGs) -fopenmp cpu_stress_no_avx.cpp -o cpu_stress_no_avx 

clean: 
	rm -f 04-dot-product cpu_stress_avx cpu_stress_no_avx  512_cpu_stress_avx
