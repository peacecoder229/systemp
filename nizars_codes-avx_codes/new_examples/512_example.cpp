#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define LOOP_NUMBER 100000000

using namespace std;
using namespace std::chrono;

int vec_width = 8;



double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int sum_no_vector(double* A, double *B, double * ret_array ){

for(int j=0; j<LOOP_NUMBER; j++){
for(int i = 0; i < vec_width; i++){
 double sum[vec_width];
 double Mult[vec_width];
 //printf("i is %d and vev_wdith = %d\n",i,vec_width ); 
 sum[i]= A[i]+B[i];
 Mult[i]= A[i]*B[i];
 ret_array[i]=sum[i]+Mult[i];
 }
}
return 1;
}




int sum_avx_vector(  double A_in[] , double B_in[] , double ret_array[] ){


double A[vec_width], B[vec_width], C[vec_width],  sum[vec_width], Mult[vec_width] ;
for (int i = 0 ; i<8 ; i++)
{
 A[i]= A_in[i];
 B[i]= B_in[i];
}
for(int j=0;j<LOOP_NUMBER;j++ ){
__m512d A_vec = _mm512_load_pd(A);
__m512d B_vec = _mm512_load_pd(B);
__m512d sum_vec , C_vec, Mult_vec ;  // = _mm512_load_pd(ret_array);
sum_vec = _mm512_add_pd(A_vec,B_vec);
Mult_vec = _mm512_mul_pd(A_vec, B_vec);
C_vec =  _mm512_add_pd(sum_vec,Mult_vec);
_mm512_store_pd(C,C_vec);
}
for(int i =0 ; i< 8 ; i++ )
{
	ret_array[i] = C[i];
}
return 1;
}


int main(){
double A[vec_width], B[vec_width], C[vec_width],D[vec_width] ;
for (int i = 0 ; i<8 ; i++)
{
 A[i]= fRand(1,100);
 B[i]= fRand(1,100);
 C[i]=0.0; D[i]=0.0;
}
high_resolution_clock::time_point t1 = high_resolution_clock::now();

//printf("Starting a new vector\n ");
sum_no_vector(A,B,C);

high_resolution_clock::time_point t2 = high_resolution_clock::now();
auto duration = duration_cast<microseconds>( t2 - t1 ).count();
cout << " regular function call: " << duration <<endl;


//printf("finished no vector\n");
high_resolution_clock::time_point t3 = high_resolution_clock::now();

sum_avx_vector(A,B,D);
//printf("finished with explicit  vector\n" );
high_resolution_clock::time_point t4 = high_resolution_clock::now();
auto duration2 = duration_cast<microseconds>( t4 - t3 ).count();
cout << "AVX instructions result: " << duration2 <<endl;


for( int i=0; i< 8 ; i++)
{
	printf("C[%d]= %f and D[%d] = %f \n ",i,C[i],i,D[i] );
}
}
