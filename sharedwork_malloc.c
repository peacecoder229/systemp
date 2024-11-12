#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define num_threads 32
#define total_transactions 200000
#define NUM_ACCOUNTS 1000000000

typedef struct {
	pthread_mutex_t lock;
	int transaction_count;
} shared_data_t;

typedef struct {
	pthread_mutex_t lock;
	long long balance;
} account_t;


//account_t accounts[NUM_ACCOUNTS];

account_t *accounts;

shared_data_t shared_data; 

void * perform_transaction(void *arg) {
	for (int i=0 ; i < total_transactions ; i++){
		usleep(rand() % 100);

		pthread_mutex_lock(&shared_data.lock);
		shared_data.transaction_count++;
		pthread_mutex_unlock(&shared_data.lock);

	}

	return NULL;
}

void * process_account(void *arg) {
	for(int i = 0 ; i < total_transactions ; i++){
		int ac_id = rand() % NUM_ACCOUNTS;
		int amount = (rand() % 200) - 100;

		pthread_mutex_lock(&accounts[ac_id].lock);
		accounts[ac_id].balance += amount;
		pthread_mutex_unlock(&accounts[ac_id].lock);

		usleep(rand() % 100);
	}

	return NULL;
}


int main() {

struct timespec start , end;
shared_data.transaction_count = 0;


accounts = (account_t *)malloc(NUM_ACCOUNTS * sizeof(account_t));

//pthread_mutex_init(&shared_data.lock, NULL);
for(int i =0 ; i < NUM_ACCOUNTS ; i++){
	pthread_mutex_init(&accounts[i].lock, NULL);
	accounts[i].balance = 1000;
}

pthread_t threads[num_threads];





clock_gettime(CLOCK_MONOTONIC, &start);


for(int i =0; i < num_threads; i++){
	//if(pthread_create(&threads[i], NULL, perform_transaction, NULL) !=0) {
	if(pthread_create(&threads[i], NULL, process_account, NULL) !=0) {
			perror("pthread_create");
			exit(1);
			}
}

for (int i=0; i < num_threads; i++){
	pthread_join(threads[i], NULL);
}	

clock_gettime(CLOCK_MONOTONIC, &end);


double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9 ;

long long total_balance_after = 0;
for(int i =0; i < NUM_ACCOUNTS ; i++){
	total_balance_after += accounts[i].balance;
}


long long total_balance_before = (long long)NUM_ACCOUNTS*1000;

//printf("Total transaction performded %d\n" , shared_data.transaction_count);
printf("Total Balance before %lld and after is %lld\n" , total_balance_before, total_balance_after);
printf(" Duration is %.6f seconds\n", duration);

for(int i=0; i < NUM_ACCOUNTS ; i++){
	pthread_mutex_destroy(&accounts[i].lock);
}
free(accounts);

//pthread_mutex_destroy(&shared_data.lock);
return 0;
}


