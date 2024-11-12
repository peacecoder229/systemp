#include <iostream>
#include <vector>
#include <dnnl.hpp>
#include <omp.h>
#include <chrono>

void initialize_matrix(float* data, int rows, int cols, bool is_triangular) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (is_triangular) {
                data[i * cols + j] = (j <= i) ? 1.0f : 0.0f; // Lower triangular
            } else {
                data[i * cols + j] = 1.0f; // All ones
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

int main() {
    // Set the number of OMP threads
    omp_set_num_threads(4);  // Adjust as needed

    // Configure dimensions and batch size
    int matrix_size = 1024;
    int batch_size = 16;

    // Create memory descriptors for input, weights, output, and bias
    dnnl::memory::dims src_dims = {batch_size, matrix_size, matrix_size};
    dnnl::memory::dims weight_dims = {batch_size, matrix_size, matrix_size};
    dnnl::memory::dims dst_dims = {batch_size, matrix_size, matrix_size};
    dnnl::memory::dims bias_dims = {1, matrix_size};

    // Create engine and stream
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream strm(eng);

    // Create MKL-DNN memory objects
    dnnl::memory src_mem({src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory weight_mem({weight_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory dst_mem({dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory bias_mem({bias_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, eng);

    // Initialize memory with data
    initialize_matrix(static_cast<float*>(src_mem.get_data_handle()), batch_size * matrix_size, matrix_size, false); // All ones
    initialize_matrix(static_cast<float*>(weight_mem.get_data_handle()), batch_size * matrix_size, matrix_size, true); // Lower triangular
    std::fill_n(static_cast<float*>(bias_mem.get_data_handle()), matrix_size, 0.5f); // Bias with all 0.5

    // Create primitive descriptor and primitive
    dnnl::matmul::primitive_desc matmul_pd(
        eng,
        src_mem.get_desc(), 
        weight_mem.get_desc(), 
        bias_mem.get_desc(), 
        dst_mem.get_desc() 
    );
    dnnl::matmul matmul_prim(matmul_pd);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform batched matrix multiplication with bias addition
    matmul_prim.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weight_mem}, {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, dst_mem}});

    // Wait for the computation to finish
    strm.wait();

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

    // Sum and print the elements of the output matrix
    float sum = sum_matrix(dst_mem);
    std::cout << "Sum of all elements in output matrix: " << sum << std::endl;

    // ... [rest of your code]
}

