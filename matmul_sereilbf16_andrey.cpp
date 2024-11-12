#include <iostream>
#include <vector>
#include <dnnl.hpp>
#include <omp.h>
#include <chrono>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <cstring>
// Assuming bf16 is represented as a 16-bit unsigned integer
using bf16 = uint16_t;

// Basic conversion of float to bf16 (truncation, not IEEE-compliant)
bf16 float_to_bf16(float fvalue) {
    uint32_t ivalue = *reinterpret_cast<uint32_t*>(&fvalue); // Bitwise cast to int
    // Simply shift to discard lower 16 bits. This is a very simplistic approach.
    bf16 bvalue = (ivalue >> 16);
    return bvalue;
}


std::unordered_map<std::string, int> parseArgs(int argc, char* argv[]) {
    std::unordered_map<std::string, int> args;
    for (int i = 1; i < argc; i += 2) {
        std::string key(argv[i]);
        if (key[0] == '-' && i + 1 < argc) {  // Ensure that the value follows the key
            args[key] = std::atoi(argv[i + 1]);
        } else {
            throw std::runtime_error("Invalid command line arguments");
        }
    }
    return args;
}


void initialize_matrix(bf16* data, int rows, int cols, bool is_triangular , float init_value) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (is_triangular) {
                data[i * cols + j] = (j <= i) ? init_value : 0.0f; // Lower triangular
            } else {
                data[i * cols + j] = init_value; // All ones
            }
        }
    }
}

/*

void initialize_matrix(bf16* data, int rows, int cols, bool is_triangular, float init_value, int cached) {
  static std::vector<bf16> cached_data;

  // Use cached data if cached is 1 and cache is not empty
  if (cached == 1 && !cached_data.empty()) {
    // Copy cached data to the output buffer
    std::copy(cached_data.begin(), cached_data.end(), data);
    return;  // Exit early if using cache
  }

  // If not using cache (cached != 1) or cache is empty
  cached_data.resize(rows * cols);  // Allocate space for cached data

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float value = init_value;
      if (is_triangular) {
        value = (j <= i) ? init_value : 0.0f;
      }
      cached_data[i * cols + j] = float_to_bf16(value);
    }
  }
}

*/



int main(int argc, char* argv[]) {

    int batch_size, src_rows, src_cols, weight_rows, weight_cols, threads, layers, cachedata; // Declare variables outside the try block
    float init_weight=0.95f, init_data=1.95f; 
    //bfloat16_t init_weight=0.95f, init_data=1.95f; 
    	// Set the number of OMP threads
    //
    try {
        auto args = parseArgs(argc, argv);

        // Extract the values using the keys
         batch_size = args["--batch_size"];
         src_rows = args["--src_row"];
         src_cols = args["--src_col"];
         weight_rows = args["--weight_row"];
         weight_cols = args["--weight_col"];
	 threads = args["--ompthreads"];
	 layers = args["--layer"];
	 cachedata = args["--cachedata"];

        // ... [rest of your main function]
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Usage: " << argv[0] << " --batch_size <value> --src_row <value> --src_col <value> --weight_row <value> --weight_col <value>\n";
        return 1;
    }
    
    omp_set_num_threads(threads);  // Adjust as needed

    dnnl::memory::dims src_dims = {batch_size, src_rows, src_cols};
    dnnl::memory::dims weight_dims = {batch_size, weight_rows, weight_cols};
    dnnl::memory::dims dst_dims = {batch_size, src_rows, weight_cols};  // Assuming matrix multiplication compatibility


    // Create engine and stream
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream strm(eng);

    // Create MKL-DNN memory objects
    // format_tag::abc to format_tag::any especially for weights
    dnnl::memory src_mem({src_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory weight_mem({weight_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory dst_mem({dst_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng);

    //
    //initialize_matrix(static_cast<bf16*>(weight_mem.get_data_handle()), batch_size * weight_rows, weight_cols, true, init_weight, cachedata); // Lower triangular
    //initialize_matrix(static_cast<bf16*>(src_mem.get_data_handle()), batch_size * src_rows, src_cols, false, init_data, cachedata); // All ones
    initialize_matrix(static_cast<bf16*>(weight_mem.get_data_handle()), batch_size * weight_rows, weight_cols, true, init_weight); // Lower triangular
    initialize_matrix(static_cast<bf16*>(src_mem.get_data_handle()), batch_size * src_rows, src_cols, false, init_data); // All ones
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int layer =0; layer < layers; ++layer){

	    // below weight matrix is copied from cached value is cacheddata option is enabled or else it is upadted measuring impact of 
	    // cache on matmul operation

    	    //initialize_matrix(static_cast<bf16*>(weight_mem.get_data_handle()), batch_size * weight_rows, weight_cols, true, init_weight, cachedata); // Lower triangular
    // Create primitive descriptor and primitive
	    dnnl::matmul::primitive_desc matmul_pd(
		eng,
		src_mem.get_desc(),
		weight_mem.get_desc(),
		dnnl::memory::desc(),  // Empty descriptor for bias
		dst_mem.get_desc()
	    );
	    dnnl::matmul matmul_prim(matmul_pd);

    // Start timing
	    //std::cout << "Momemt before execution " << std::endl;
    // Perform batched matrix multiplication without bias addition
    	   matmul_prim.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weight_mem}, {DNNL_ARG_DST, dst_mem}});

    // Wait for the computation to finish
           strm.wait();

	auto src_ptr = static_cast<bf16*>(src_mem.get_data_handle());
	auto dst_ptr = static_cast<bf16*>(dst_mem.get_data_handle());
	size_t size = batch_size * src_rows * src_cols * sizeof(bf16); // Calculate the buffer size
	memcpy(dst_ptr, src_ptr, size);



    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "matmultime=" << elapsed.count() << "seconds." << std::endl;

}

