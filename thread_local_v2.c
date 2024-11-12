#include <pthread.h>
#include <stdio.h>

#define NUM_THREADS 4

// Global sum variable to store the result
int global_sum = 0;

// Thread-local sum variable to store intermediate results
__thread int thread_local_sum = 0;

// Function that will be executed by each thread
void* add_numbers(void *arg) {
  int i;
  int *numbers = (int *)arg;
  int n = numbers[0];
  int start = numbers[1];

  // Calculate the sum of the first n numbers
  for (i = start; i <= n+start; i++) {
    thread_local_sum += i;
  }

  // Add the thread's intermediate result to the global sum
  global_sum += thread_local_sum;

  return NULL;
}

int main() {
  pthread_t threads[NUM_THREADS];
  int numbers[NUM_THREADS][2] = {{100, 1}, {100, 101}, {100, 201}, {100, 301}};
  int i, rc;

  // Create NUM_THREADS threads
  for (i = 0; i < NUM_THREADS; i++) {
    rc = pthread_create(&threads[i], NULL, add_numbers, (void *)numbers[i]);
    if (rc) {
      printf("Error creating thread %d\n", i);
      return 1;
    }
  }

  // Wait for all threads to finish
  for (i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  // Print the final result
  printf("The sum of the first %d numbers is %d\n",
         NUM_THREADS * 100, global_sum);

  return 0;
}

