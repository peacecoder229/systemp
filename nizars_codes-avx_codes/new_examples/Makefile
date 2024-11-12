

all: clean 512_example   with_no_avx

512_example:
	clang++ 512_example.cpp -o 512_example -mavx512f -mavx512er -mavx512pf -mavx512cd -g 	
with_no_avx:
	clang++ no_avx.cpp -o no_avx -O3 -g # -fsave-optimization-record

clean:
	rm -f 512_example no_avx
