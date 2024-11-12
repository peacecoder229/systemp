#include <omp.h> // for omp functions and pragma
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <errno.h>   // for errno
#include <limits.h>  // for INT_MAX
#include <stdlib.h>  // for strtol
#include <ctime> //To measure time
#include <sys/time.h>
#include <math.h>

const int N = 83040;

double slow_dot_product(const double *a, const double *b) {
  double answer = 0.0;
  for(int ii = 0; ii < N; ++ii)
    answer += a[ii]*b[ii];
  return answer;
}

using namespace std;

int test_function()
{
int d = 0;
for(int j =0; j<200000;j++)
        {
          int a =j;
          int b =j+2;
          int c = a+b *(a*b);
          d += a*a+b*b+c*c;

        }
return d;
}
float res_array[112];
int main( int argc, char **argv){
int nthreads, tid;
char *p;
int num;
long conv;
double delta, answer;
struct timeval start, end;
__attribute__ ((aligned (32))) double a[N], b[N]; 
 
   for(int ii = 0; ii < N; ++ii) 
     a[ii] = b[ii] = ii/sqrt(N); 

gettimeofday(&start, NULL);
// read command line input, if available read the first input as integer and use it to set the number of threads 
// if no input is given we allow a dynamic number of threads to be set by the kernel
if (argc > 1)
	{	conv = strtol(argv[1], &p, 10); //read the input from the command line
		num = conv; // transform to int
		omp_set_num_threads(num); // reques that the number of requested threads is the number we put in
	}
//omp_set_dynamic(0);     // Explicitly disable dynamic teams
/* Fork a team of threads giving them their own copies of variables */
	#pragma omp parallel private(nthreads, tid)
	{
	tid = omp_get_thread_num();
	if(tid == 0)
	{
	int N= omp_get_num_threads();
	printf("the total number of threads is %d \n",N);
	}
	#pragma omp for
	for (int i=1; i<50000000 ; i++) {
	answer = slow_dot_product(a , b );
	}
	}
gettimeofday(&end, NULL);
delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
         end.tv_usec - start.tv_usec) / 1.e6;

printf("the time spent %f\n, answer %f \n",delta,answer);
return 0;
}
