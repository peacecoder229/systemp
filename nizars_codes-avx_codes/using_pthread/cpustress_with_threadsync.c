// Same as futex-basic-process.c, but using threads.
//
// Main differences:
// 1. No need for shared memory calls (shmget, etc.), since threads share the
//    address space we can use a simple pointer.
// 2. Use the _PRIVATE versions of system calls since these can be more
//    efficient within a single process.
//
// Eli Bendersky [http://eli.thegreenplace.net]
// This code is in the public domain.
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>
#include <math.h>

const int N = 83040;
// The C runtime doesn't provide a wrapper for the futex(2) syscall, so we roll
// our own.

//rdtsc used to count no of PCU cycles

int threadno = 1;
static int shared_data =0;
const int sign = 100;
pthread_mutex_t lock;
__attribute__ ((aligned (32))) double a[N], b[N];

static long long  read_tsc(void){

    register unsigned long eax1, edx1;
    asm volatile("rdtsc; lfence; \n"
            : "=a"(eax1), "=d"(edx1)
            :
            :
            );
    unsigned long long tsc_output = (edx1<<32)|eax1;
    printf("%lld, ", tsc_output);
    //printf("%ld, %ld", edx1, eax1);
    return ((edx1<<32)|eax1);
}


int test_function()
{
int d = 0;
for(int j =0; j<2000000000;j++)
        {
          int a =j;
          int b =j+2;
          int c = a+b *(a*b);
          d += a*a+b*b+c*c;

        }
return d;
}

double dot_product(const double *a, const double *b) {
  __m512d sum_vec = _mm512_set_pd(0.0, 0.0, 0.0, 0.0 , 0.0,0.0,0.0,0.0 ); //set 256 avx varaible with 4 doubles equal to zerpo
  double final;
  /* Add up partial dot-products in blocks of 256 bits */
  for(int ii = 0; ii < N/8; ++ii) {
    __m512d x = _mm512_load_pd(a+8*ii); //load variaible
    __m512d y = _mm512_load_pd(b+8*ii); //load varaible
    __m512d z = _mm512_mul_pd(x,y); // multiply vector
    sum_vec = _mm512_add_pd(sum_vec, z); //sum original vector with multiplication results
  }

  /* Find the partial dot-product for the remaining elements after
 *  *    * dealing with all 256-bit blocks. */
 // for(int ii = N-N%8; ii < N; ++ii)
       final = _mm512_reduce_add_pd(sum_vec);
        return  final;
  }

double reduce_vector2(__m256d input) {
  __m256d temp = _mm256_hadd_pd(input, input);
  __m128d sum_high = _mm256_extractf128_pd(temp, 1); //get the highest 128 bits
  __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(temp)); // add the higest 128 bit with the lowerst 128 bits
  return ((double*)&result)[0];
}

double dot_product_256(const double *a, const double *b) {
  __m256d sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0); //set 256 avx varaible with 4 doubles equal to zerpo

  /* Add up partial dot-products in blocks of 256 bits */
  for(int ii = 0; ii < N/4; ++ii) {
    __m256d x = _mm256_load_pd(a+4*ii); //load variaible
    __m256d y = _mm256_load_pd(b+4*ii); //load varaible
    __m256d z = _mm256_mul_pd(x,y); // multiply vector
    sum_vec = _mm256_add_pd(sum_vec, z); //sum original vector with multiplication results
  }

  /* Find the partial dot-product for the remaining elements after
 *  *    * dealing with all 256-bit blocks. */
  double final = 0.0;
  for(int ii = N-N%4; ii < N; ++ii)
    final += a[ii] * b[ii];

  return reduce_vector2(sum_vec) + final;
}

double slow_dot_product(const double *a, const double *b) {
  double answer = 0.0;
  for(int ii = 0; ii < N; ++ii)
    answer += a[ii]*b[ii];
  return answer;
}

//futex syscall wrapper function
//
int futex(int* uaddr, int futex_op, int val, const struct timespec* timeout,
          int* uaddr2, int val3) {
  return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}

// Waits for the futex at futex_addr to have the value val, ignoring spurious
// wakeups. This function only returns when the condition is fulfilled; the only
// other way out is aborting with an error.
// wait on futex address till its value is equal to the value supplied

void wait_on_futex_value(int* futex_addr, int val) {
  while (1) {
    int futex_rc = futex(futex_addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
    if (futex_rc == -1) {
      if (errno != EAGAIN) {
        perror("futex");
        exit(1);
      }
    } else if (futex_rc == 0) {
      if (*futex_addr == val) {
        // This is a real wakeup.
        return;
      }
    } else {
      abort();
    }
  }
return;
}

// A blocking wrapper for waking a futex. Only returns when a waiter has been
// woken up.waking number of threads as specified by wakeno from the main thread
//  does not return till no of threads woken is equal to specified no by wakeno
int wake_futex_blocking(int* futex_addr, int wakeno ) {
	int futex_rc_tot = 0;
  while (1) {
    int futex_rc = futex(futex_addr, FUTEX_WAKE_PRIVATE, wakeno, NULL, NULL, 0);
    futex_rc_tot = futex_rc_tot + futex_rc;
    if (futex_rc == -1) {
      perror("futex wake");
      exit(1);
    } 
    else if ((futex_rc > 0)&&( futex_rc_tot == wakeno)) {
	    return futex_rc_tot;
    }
    else
      continue;
    }
  return 0;
}


void * threadfunc(void* ) {
  int s;
  //printf("child waiting for A\n");
  //wait_on_futex_value(shared_data, 0xA);

  // Write 0xB to the shared data and wake up parent.

  s = pthread_mutex_lock(&lock);
  if(s != 0) {
	  printf("Lock Error %d\n", s);
  }
	 

  printf("child incrementing shared variable with current value = %d\n", shared_data);
  printf("Shared data is %d\n", shared_data);
  shared_data = shared_data + 1;
  printf("Shared data is %d\n", shared_data);

  if(shared_data == threadno){
  wake_futex_blocking(&shared_data, 1);
  }
  pthread_mutex_unlock(&lock);
  
  if(s != 0) {
	  printf("Lock Error %d\n", s);
  }
 /* if(*shared_data == threadno){
  wake_futex_blocking(shared_data, 1);
  } */

  printf("child waiting for GO CMD\n");
  wait_on_futex_value(&shared_data, sign);
  double res = dot_product(a,b);
  res += dot_product_256(a,b);
  res += slow_dot_product(a,b);
  printf("CPU stress DONE\n");
  return NULL;
}

int main(int argc, char** argv) {
  int s;
  unsigned long long int t1, t2, t3;
  pthread_t childt[threadno];
  int i;
  // __attribute__ ((aligned (32))) double a[N], b[N];

   for(int ii = 0; ii < N; ++ii)
     a[ii] = b[ii] = ii/sqrt(N);

  t1 = read_tsc();
  //static int threadno = 1;
  //
  //
  //Initialize mutex
  
  pthread_mutexattr_t mtxattr;
  pthread_mutexattr_settype(&mtxattr, PTHREAD_MUTEX_ERRORCHECK);
  //pthread_mutexattr_settype(&mtxattr, PTHREAD_MUTEX_DEFAULT);
  if (pthread_mutex_init(&lock, &mtxattr) != 0)
    {
        printf("\n mutex init failed\n");
        return 1;
    }

  if(argc > 1){
	  threadno = atoi(argv[1]);
	  printf("No of requested threads are %d\n", threadno);
  }
  for (i=0;i<threadno;i++){
        printf("shared data addr %d and value is %d \n", &shared_data, shared_data);
  	pthread_create(&childt[i], NULL, threadfunc, NULL);
  }

  // Parent thread.

  printf("parent launched the threads and waiting for all of them to increment the sharevalue\n");
  // Write 0xA to the shared data and wake up child.
  //*shared_data = 0xA;
  //wake_futex_blocking(shared_data);

  //printf("parent waiting for B\n");
  t2 = read_tsc();
  wait_on_futex_value(&shared_data, threadno);
  s = pthread_mutex_lock(&lock);
    if(s != 0) {
	              printf("Lock Error %d\n", s);}

  shared_data = sign;
  
   pthread_mutex_unlock(&lock);

     if(s != 0) {
	               printf("Lock Error %d\n", s);
		         }

  printf("Shared data changed within parent and it is now %d\n", shared_data);
  int woken = wake_futex_blocking(&shared_data, threadno);
  printf("All %d threads are  told to GO\n and wake call resulted in =%d no of threads woken\n", threadno, woken);
  t3 = read_tsc();
  printf("Time Threads starts = %llu\n All threads are launched = %llu\n and Threads are synced %llu\n", t1, t2, t3);
  printf("time of launch cycle is %llu and time for sync-back is %llu\n", (t2-t1), (t3-t1)); 

  for(i=0;i<threadno;i++){
	printf("Attempting to join thread id = %ld and iteration is %d\n", childt[i], i);
	usleep(1);
  	pthread_join(childt[i], NULL);
  	//printf("joined thread id = %ld\n", childt[i]);
  }

printf("DONE\n");  
pthread_mutex_destroy(&lock);
}
