all:  clean cpustress_with_threadsync
CC=g++ -std=c++11 -mavx2 -fopenmp -g
ICBC=/opt/intel/compilers_and_libraries_2018.3.222/linux/bin/intel64/icpc 
FLAGs=-std=c++11 -mavx2 -fopenmp -g
cpustress_with_threadsync:
#	$(ICBC) -Wall -g -o cpustress_with_threadsync cpustress_with_threadsync.c -lpthread -fpermissive -xCOMMON-AVX512 
	$(CC) -Wall -g -o cpustress_with_threadsync_gcc cpustress_with_threadsync.c -lpthread -fpermissive -xCOMMON-AVX512 
clean: 
	rm -f cpustress_with_threadsync
