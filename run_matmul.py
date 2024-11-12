import subprocess
import math
import time
import re
def calculate_dimensions(src_mem_kb, skew, bs):
    src_col = math.sqrt(src_mem_kb * 1024 / (2 * skew * bs)) 
    src_row = src_col * skew
    weight_row = src_col
    weight_col = weight_row * skew
    #total_mem_mb = (src_col * src_row + weight_row * weight_col + src_row * weight_col) * 4 * bs / 1024 / 1024
    total_mem_mb = (src_col * src_row + weight_row * weight_col + src_row * weight_col) * 2 * bs / 1024 / 1024 # bf16 2 bytes
    return int(src_row), int(src_col), int(weight_row), int(weight_col), total_mem_mb

def run_matmul(src_row, src_col, weight_row, weight_col, batch_size, threads, layer, cacheinput):
    core = threads - 1
    #cmd = f"numactl -C 0-{core} ./matmul_args --src_row {src_row} --src_col {src_col} " \
    cmd = f"numactl -C 0-{core} ./matmul_bf16 --src_row {src_row} --src_col {src_col} " \
          f"--weight_row {weight_row} --weight_col {weight_col} " \
          f"--batch_size {batch_size} --ompthreads {threads} --layer {layer} --cachedata {cacheinput}"
    print(cmd)
    time.sleep(1)
    #result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    output = result.stdout
    matches = re.findall(r'matmultime=([\d.]+)seconds', output)
    if matches:
                     #time_str = match.group(1)
        time_str = matches[0]# This will be the string '0.645657' if the line is found
        print(f"Elapsed time: {time_str} seconds")
    else:
        print("Elapsed time not found in the output.")

    #time_str = output.split("Elapsed time: ")[1].split(" seconds")[0]
    return float(time_str)

def main():
    src_mem_sizes_kb = [102400]  # in KB
    #src_mem_sizes_kb = [512, 6144, 12288, 102400, 524288]  # in KB
    #src_mem_sizes_kb = [256, 512, 5024, 100048, 512000]
    ompthreads_values = [24]
    skew_values = [1]
    batch_size = 1
    layer = 80

    with open("matmul_results.csv", "a") as file:
        file.write("threads,skew,src_mem_kb,total_mem_kb,execution_time,inputdatacached\n")
        for cacheinput in ("0" , "1"):
            for src_mem_kb in src_mem_sizes_kb:
                for threads in ompthreads_values:
                    for skew in skew_values:
                        src_row, src_col, weight_row, weight_col, total_mem_mb = calculate_dimensions(src_mem_kb, skew, batch_size)
                        execution_time = run_matmul(src_row, src_col, weight_row, weight_col, batch_size, threads, layer, cacheinput)
                        print("threads,skew,src_mem_kb,total_mem_mb,execution_time,cacheinput\n")
                        print(f"{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{cacheinput}\n")
                        file.write(f"{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{cacheinput}\n")

if __name__ == "__main__":
    main()

