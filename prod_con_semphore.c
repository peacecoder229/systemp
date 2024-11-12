#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define NUM_SLOTS 5

sem_t empty_slots;
sem_t full_slots;
int buffer[NUM_SLOTS];
int buffer_index = 0;

// Parameters for controlling production and consumption
int num_items_to_produce = 0;
int num_items_to_consume = 0;

int produce_item() {
    // Simulate producing an item (replace with actual logic)
    int item = rand() % 100;  // Generate a random item
    printf("Producer produced item: %d\n", item);
    sleep(1);  // Simulate production time
    return item;
}

void consume_item(int item) {
    // Simulate consuming the item (replace with actual logic)
    printf("Consumer consumed item: %d\n", item);
    sleep(2);  // Simulate consumption time
}

void *producer(void *arg) {
    for (int i = 0; i < num_items_to_produce; ++i) {
        int item = produce_item();

        sem_wait(&empty_slots);  // Wait for an empty slot

        buffer[buffer_index] = item;
        buffer_index = (buffer_index + 1) % NUM_SLOTS;

        sem_post(&full_slots);  // Signal a filled slot
    }
    return NULL;
}

void *consumer(void *arg) {
    for (int i = 0; i < num_items_to_consume; ++i) {
        sem_wait(&full_slots);  // Wait for a filled slot

        int item = buffer[buffer_index];
        buffer_index = (buffer_index - 1 + NUM_SLOTS) % NUM_SLOTS;

        sem_post(&empty_slots);  // Signal an empty slot

        consume_item(item);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_items_to_produce> <num_items_to_consume>\n", argv[0]);
        return 1;
    }

    num_items_to_produce = atoi(argv[1]);
    num_items_to_consume = atoi(argv[2]);

    // Initialize semaphores
    sem_init(&empty_slots, 0, NUM_SLOTS);
    sem_init(&full_slots, 0, 0);

    pthread_t producer_thread, consumer_thread;

    pthread_create(&producer_thread, NULL, producer, NULL);
    pthread_create(&consumer_thread, NULL, consumer, NULL);

    pthread_join(producer_thread, NULL);
    pthread_join(consumer_thread, NULL);

    return 0;
}

