#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#define NUM_THREADS 32
#define NUM_FILES 10000
#define FILE_SIZE 1024 * 1024 // 1 MB
#define BUFFER_SIZE 4096 * 16 // 4 KB
#define NUM_OPERATIONS 10000

typedef struct {
    int thread_id;
} thread_data_t;

void *file_operation(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    char buffer[BUFFER_SIZE];
    int file_index;
    ssize_t bytes_read, bytes_written;
    
    for (int i = 0; i < NUM_OPERATIONS; i++) {
        file_index = rand() % NUM_FILES;
        char filename[256];
        sprintf(filename, "/home/exp/file_%d.dat", file_index);
        
        int fd = open(filename, O_RDWR);
        if (fd == -1) {
            perror("open");
            continue;
        }
        
        // Read operation
        bytes_read = pread(fd, buffer, BUFFER_SIZE, rand() % (FILE_SIZE - BUFFER_SIZE));
        if (bytes_read == -1) {
            perror("pread");
            close(fd);
            continue;
        }
        
        // Modify buffer to simulate processing
        for (int j = 0; j < bytes_read; j++) {
            buffer[j] = ~buffer[j];
        }
        
        // Write operation
        bytes_written = pwrite(fd, buffer, bytes_read, rand() % (FILE_SIZE - BUFFER_SIZE));
        if (bytes_written == -1) {
            perror("pwrite");
        }
        
        close(fd);
    }

    return NULL;
}

int main() {
    srand(time(NULL));

    // Initialize files with random data
    char buffer[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = rand() % 256;
    }

    for (int i = 0; i < NUM_FILES; i++) {
        char filename[256];
        sprintf(filename, "/home/exp/file_%d.dat", i);
        
        int fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd == -1) {
            perror("open");
            exit(1);
        }
        
        for (int j = 0; j < FILE_SIZE / BUFFER_SIZE; j++) {
            if (write(fd, buffer, BUFFER_SIZE) != BUFFER_SIZE) {
                perror("write");
                close(fd);
                exit(1);
            }
        }
        
        close(fd);
    }

    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create threads for file operations
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        if (pthread_create(&threads[i], NULL, file_operation, (void *)&thread_data[i]) != 0) {
            perror("pthread_create");
            exit(1);
        }
    }

    // Wait for threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("File operations completed in %.6f seconds\n", duration);

    return 0;
}

