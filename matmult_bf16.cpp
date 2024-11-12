#include <iostream>
#include <vector>
#include <dnnl.hpp>
#include <omp.h>
#include <chrono>
#include <string>
#include <unordered_map>


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




void initialize_matrix(float* data, int rows, int cols, bool is_triangular , float init_value) {
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

float sum_matrix(const dnnl::memory& mem) {
    float sum = 0.0f;
    const float* data = static_cast<float*>(mem.get_data_handle());
    for (size_t i = 0; i < mem.get_desc().get_size() / sizeof(float); ++i) {
        sum += data[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {

    int batch_size, src_rows, src_cols, weight_rows, weight_cols, threads, layers; // Declare variables outside the try block
    float init_weight=0.95f, init_data=1.95f; 
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
    dnnl::memory src_mem({src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory weight_mem({weight_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory dst_mem({dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, eng);

    initialize_matrix(static_cast<float*>(weight_mem.get_data_handle()), batch_size * weight_rows, weight_cols, true, init_weight); // Lower triangular
    // Initialize memory with data
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int layer =0; layer < layers; ++layer){
	    init_data = init_data*2;
    	    initialize_matrix(static_cast<float*>(src_mem.get_data_handle()), batch_size * src_rows, src_cols, false, init_data); // All ones

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
	   //std::cout << "Momemt after execution " << std::endl;
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

    // Sum and print the elements of the output matrix
    //float sum = sum_matrix(dst_mem);
    //std::cout << "Sum of all elements in output matrix: " << sum << std::endl;

    // ... [rest of your code]
}

