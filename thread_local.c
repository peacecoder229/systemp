#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 5

pthread_key_t key;

void *calculate_thread_sum(void *threadid) {
  int i, sum = 0;
  int *my_sum = (int *) malloc(sizeof(int));

  for (i = 0; i < 100; i++) {
    sum += i;
  }

  *my_sum = sum;
  pthread_setspecific(key, (void *) my_sum);

  pthread_exit(NULL);
}

int main(void) {
  pthread_t threads[NUM_THREADS];
  int rc, t;

  pthread_key_create(&key, NULL);

  for (t = 0; t < NUM_THREADS; t++) {
    rc = pthread_create(&threads[t], NULL, calculate_thread_sum, (void *) t);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  for (t = 0; t < NUM_THREADS; t++) {
    rc = pthread_join(threads[t], NULL);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
    int *thread_sum = (int *) pthread_getspecific(key);
    printf("Thread %d sum is %d\n", t, *thread_sum);
  }

  pthread_key_delete(key);
  pthread_exit(NULL);
}

