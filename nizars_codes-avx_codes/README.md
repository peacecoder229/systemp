# Introduction

this code implement two vector sum addition using sum, there are multiple generated binary.

There are 4 folders available

the AVX2-example implement a sum of two array using avx2 instructions, this should be compatible with Haswell and newer processors. This code is received from another gut repo. The implementation of the code using intel intrinsic

the new example folder contains a same code implemented using avx3 intructions, some modification had to be made like dividing doubles into groups of 8 instead of 4 and adding divide and conquer add.

Looped_code: this code stres the CPU using avx 2 an avx 3 instructions. it implement the previous in a loop using open openMP. 

To run this code you can either build hte binary or use the precompiled binaries

note that these binaries are complied using Intel c++ compiler, which can generate AVX512 istructions

to use it with GCC compiler you need GCC verion 4.9 or newer. The compiler falg should also be edited to support AVX3 instructios with proper flags.

Note that the Makefile has the address of the Intel C++ compiler (ICPC) hardcoded and it needs to the location of ICPC binary to run this code

	cd looped_code
	./512_cpu_stress_avx # this will divide the code to as many threads avaialble on the system
	./512_cpu_stress_avx 25 # this will do the AVX512 stress code using 25 threads
	./cpu_stress_avx # stresss test using 256 avx
	./cpu_stress_no_avx # stres test without AVX (most compiler will attemp using SSE 128 bit instrucitns)
	./04-dot-product # this will do the same without using any vector insructions

the make command might need change if avx512 to be used. 

make sure to use gcc 4.9 or newer for avx512 or intel icpc compiler

skylake-x and newer intel IA cpu support AVX3 or AVX256 these instructions allow to operate 8 double operation or 16 float

NOtes:

Some compiler can automatically vectorize code when running in a loop and use AVX or SSE instructions

we can verify if this code is using avx instructions with emon data colleciton or by using dumpelf and checking the binary for registers or instrctions that use AVX.

for more information about the AVX instcutions and expermoinet please refered to the docx file in the repo

