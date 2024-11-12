#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ROW_A 3
#define COL_A 3
#define ROW_B 3
#define COL_B 3

int A[ROW_A][COL_A];
int B[ROW_B][COL_B];
int C[ROW_A][COL_B];

struct parameters {
    int i;
    int j;
};

void *compute_element(void *params) {
    struct parameters *p = (struct parameters *) params;
    int sum = 0;
    for (int k = 0; k < COL_A; k++) {
        sum += A[p->i][k] * B[k][p->j];
    }
    C[p->i][p->j] = sum + A[p->i][p->j];
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[ROW_A][COL_B];
    int rc;

    // Initialize matrices A and B
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_A; j++) {
            A[i][j] = i + j;
        }
    }
    for (int i = 0; i < ROW_B; i++) {
        for (int j = 0; j < COL_B; j++) {
            B[i][j] = i * j;
        }
    }

    // Create threads to compute each element in matrix C
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            struct parameters *p = (struct parameters *) malloc(sizeof(struct parameters));
            p->i = i;
            p->j = j;
            rc = pthread_create(&threads[i][j], NULL, compute_element, (void *) p);
            if (rc) {
                printf("Error creating thread\n");
                exit(-1);
            }
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            rc = pthread_join(threads[i][j], NULL);
            if (rc) {
                printf("Error joining thread\n");
                exit(-1);
            }
        }
    }

    // Print result
    for (int i = 0; i < ROW_A; i++) {
        for (int j = 0; j < COL_B; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    pthread_exit(NULL);
    return 0;
}

