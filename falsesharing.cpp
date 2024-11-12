#include <iostream> 
#include <thread>
#include <ctime>
#include <vector>
struct Counter {
    int thread_id;
    unsigned long long counter;
    char padding[48];
};

void increment_counter(Counter* counter, unsigned long long  num_count) {
    for (unsigned long long i = 0; i < num_count; ++i) {
        counter->counter++;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_count>\n";
        return 1;
    }

    struct timespec start, end;
    printf("Size of Counter struct: %lu bytes\n", sizeof(struct Counter));
    //int num_count = std::stoi(argv[1]);
    unsigned long long num_count = std::stoull(argv[1]);
    const int num_threads = 10;
    Counter counters[num_threads];
    for(int i =0 ; i< num_threads; i++){
	    counters[i].thread_id = i;
    }
    //counters[0].thread_id = 0;
    //counters[1].thread_id = 1;
    std::vector<std::thread> threads;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i =0 ; i< num_threads; i++){
    //std::thread thread1(increment_counter, &counters[i], num_count);
    threads.emplace_back(increment_counter, &counters[i], num_count);
    }
    //std::thread thread2(increment_counter, &counters[1], num_count);

    //clock_gettime(CLOCK_MONOTONIC, &end);
    //thread1.join();
    //thread2.join();
    for(auto &thread : threads){
	    thread.join();
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;

    std::cout << "Time taken is " << duration << std::endl ;

    return 0;
}

