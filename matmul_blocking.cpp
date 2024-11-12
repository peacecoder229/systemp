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

void initialize_matrix(bf16* data, int rows, int cols, bool is_triangular, float init_value) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (is_triangular) {
                data[i * cols + j] = (j <= i) ? float_to_bf16(init_value) : 0; // Lower triangular
            } else {
                data[i * cols + j] = float_to_bf16(init_value); // All ones
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int batch_size, src_rows, src_cols, weight_rows, weight_cols, threads, layers, cachedata, blocksize; // Declare variables outside the try block
    float init_weight = 0.95f, init_data = 1.95f;

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
        blocksize = args["--blocksize"];
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Usage: " << argv[0] << " --batch_size <value> --src_row <value> --src_col <value> --weight_row <value> --weight_col <value> --ompthreads <value> --layer <value> --cachedata <value> --blocksize <value>\n";
        return 1;
    }

    omp_set_num_threads(threads);  // Adjust as needed

    dnnl::memory::dims src_dims = {batch_size, src_rows, src_cols};
    dnnl::memory::dims weight_dims = {batch_size, weight_rows, weight_cols};
    dnnl::memory::dims dst_dims = {batch_size, src_rows, weight_cols};  // Assuming matrix multiplication compatibility

    // Create engine and stream
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream strm(eng);

    // Create oneDNN memory objects
    dnnl::memory src_mem({src_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng);
    dnnl::memory weight_mem({weight_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::acb}, eng); // Column-major for weight
    dnnl::memory dst_mem({dst_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng);

    // Initialize matrices
    initialize_matrix(static_cast<bf16*>(weight_mem.get_data_handle()), batch_size * weight_rows, weight_cols, true, init_weight); // Lower triangular
    initialize_matrix(static_cast<bf16*>(src_mem.get_data_handle()), batch_size * src_rows, src_cols, false, init_data); // All ones

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int layer = 0; layer < layers; ++layer) {
        // Create primitive descriptor and primitive
        dnnl::matmul::primitive_desc matmul_pd(
            eng,
            src_mem.get_desc(),
            weight_mem.get_desc(),
            dnnl::memory::desc(),  // Empty descriptor for bias
            dst_mem.get_desc()
        );
        dnnl::matmul matmul_prim(matmul_pd);

        // Perform block-based batched matrix multiplication
        for (int i = 0; i < batch_size; i += blocksize) {
            int block_end = std::min(i + blocksize, batch_size);
            for (int j = 0; j < src_rows; j += blocksize) {
                int row_block_end = std::min(j + blocksize, src_rows);
                for (int k = 0; k < weight_cols; k += blocksize) {
                    int col_block_end = std::min(k + blocksize, weight_cols);

                    dnnl::memory::dims block_src_dims = {block_end - i, row_block_end - j, src_cols};
                    dnnl::memory::dims block_weight_dims = {block_end - i, weight_rows, col_block_end - k};
                    dnnl::memory::dims block_dst_dims = {block_end - i, row_block_end - j, col_block_end - k};

                    dnnl::memory block_src_mem({block_src_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng, static_cast<bf16*>(src_mem.get_data_handle()) + i * src_rows * src_cols + j * src_cols);
                    dnnl::memory block_weight_mem({block_weight_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::acb}, eng, static_cast<bf16*>(weight_mem.get_data_handle()) + i * weight_rows * weight_cols + k * weight_cols);
                    dnnl::memory block_dst_mem({block_dst_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc}, eng, static_cast<bf16*>(dst_mem.get_data_handle()) + i * src_rows * weight_cols + j * weight_cols);

                    dnnl::matmul::primitive_desc block_matmul_pd(
                        eng,
                        block_src_mem.get_desc(),
                        block_weight_mem.get_desc(),
                        dnnl::memory::desc(),  // Empty descriptor for bias
                        block_dst_mem.get_desc()
                    );
                    dnnl::matmul block_matmul_prim(block_matmul_pd);

                    block_matmul_prim.execute(strm, {{DNNL_ARG_SRC, block_src_mem}, {DNNL_ARG_WEIGHTS, block_weight_mem}, {DNNL_ARG_DST, block_dst_mem}});

                    strm.wait();
                }
            }
        }
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "matmultime=" << elapsed.count() << " seconds." << std::endl;

    return 0;
}

