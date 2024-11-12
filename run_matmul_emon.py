import subprocess
import argparse
import math
import time

def calculate_dimensions(src_mem_kb, skew, bs):
    src_col = math.sqrt(src_mem_kb * 1024 / 2 / skew / bs) 
    src_row = src_col * skew
    weight_row = src_col
    weight_col = weight_row * skew
    #total_mem_mb = (src_col * src_row + weight_row * weight_col + src_row * weight_col) * 4 * bs / 1024 / 1024
    total_mem_mb = (src_col * src_row + weight_row * weight_col + src_row * weight_col) * 2 * bs / 1024 / 1024 # bf16 2 bytes
    return int(src_row), int(src_col), int(weight_row), int(weight_col), total_mem_mb

def run_matmul(src_row, src_col, weight_row, weight_col, batch_size, threads, layer, total_mem_mb, cacheinput, args ):
    core = 1 if (threads == 1) else (threads - 1)
    startcore = 1 if (threads == 1) else 0
    #cmd = f"numactl -C 0-{core} ./matmul_args --src_row {src_row} --src_col {src_col} " \
    #cmd = f"numactl -C 0-{core} ./matmul_bf16 --src_row {src_row} --src_col {src_col} " \
    if args.dnnverbose:
        env = f"ONEDNN_VERBOSE=1 "
    elif args.dnnverbose and args.noamx:
        env = f"ONEDNN_VERBOSE=1 ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 "
    else:
        env = f""

    env = f""
    if args.noamx:
        env = f"ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16"
    if args.dnnverbose:
        if env:  # Checking if env is not empty
            env += " ONEDNN_VERBOSE=1"
        else:
            env = "ONEDNN_VERBOSE=1"

    #env = "ONEDNN_VERBOSE=1 ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 "
    cmd = f"{env} numactl -C {startcore}-{core} ./matmul_sereilbf16 --src_row {src_row} --src_col {src_col} " \
          f"--weight_row {weight_row} --weight_col {weight_col} " \
          f"--batch_size {batch_size} --ompthreads {threads} --layer {layer} --cachedata {cacheinput}"
    print(cmd)
    mem_mb = round(total_mem_mb, 1)
    mem_kb = round(total_mem_mb*1000, 1)
    time.sleep(1)
    #new_cmd = f'tmc -T all -x rdas -i "NA" -Z metrics2 -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "thread{threads}_{mem_mb}MB_SRCrows{src_row}" -n -u -c "{cmd}"'
    #new_cmd = f'tmc -T all -x rdas -i "NA" -Z metrics2 -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "{mem_kb}kB_NoAMX{str(args.noamx).lower()}" -n -u -c "{cmd}"'
    #new_cmd = f'tmc -T all -x rdas -i "NA" -Z metrics -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "{mem_kb}kB_NoAMX{str(args.noamx).lower()}" -n -u -c "{cmd}"'
    new_cmd = f'tmc -T all -x rdas -i "NA" -Z flex -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "{mem_kb}kB_NoAMX{str(args.noamx).lower()}" -n -u -c "{cmd}"'
    print(new_cmd) 
    finalcmd = new_cmd if args.emon else cmd

    #result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    #result = subprocess.run(new_cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    result = subprocess.run(finalcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    time.sleep(5)
    output = result.stdout
    #time_str = output.split("Elapsed time: ")[1].split(" seconds")[0]
    #return float(time_str)
    #print(output)
    import re

    # Assuming result.stdout contains the output from your subprocess command

    # Use a regular expression to find and extract the elapsed time
    #match = re.search(r'Elapsed time: ([\d.]+) seconds', output)
    matches = re.findall(r'matmultime=([\d.]+)seconds', output)
    if matches:
        #time_str = match.group(1)
        time_str = matches[0]# This will be the string '0.645657' if the line is found
        print(f"Elapsed time: {time_str} seconds")
    else:
        print("Elapsed time not found in the output.")


    return float(time_str)

def main():


    import os

    command = """
    echo 3 > /proc/sys/vm/drop_caches &&
    swapoff -a &&
    swapon -a &&
    printf '\n%s\n' 'Ram-cache and Swap Cleared'
    """

    # Execute the command
    status = os.system(command)

    # Check if the command was executed successfully
    if status == 0:
        print('Ram-cache and Swap Cleared')
    else:
        print('An error occurred')


    parser = argparse.ArgumentParser(description="Run matmul with optional emon monitoring.")
    parser.add_argument('--emon', action='store_true', help="Enable emon monitoring")
    parser.add_argument('--noamx', action='store_true', help="Enable emon monitoring")
    parser.add_argument('--dnnverbose', action='store_true', help="Enable emon monitoring")
    args = parser.parse_args()
    emon = args.emon
    #src_mem_sizes_kb = [18, 32, 128, 512, 1152]  # in KB
    #src_mem_sizes_kb = [18, 32, 128]  # in KB
    #src_mem_sizes_kb = [512, 1152]  # in KB
    #src_mem_sizes_kb = [512000]  # in KB
    #src_mem_sizes_kb = [512000]  # in KB
    src_mem_sizes_kb = [18, 32, 128, 512, 5024, 100048, 512000]
    #src_mem_sizes_kb = [512, 1024, 2048, 3048, 6144]
    ompthreads_values = [32]
    #ompthreads_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    skew_values = [1]
    batch_size = 1
    #layer = 40
    #layerset = {18: 2000000,  32: 1600000, 128: 1000000, 512: 1000000, 5024: 40000, 100048: 16800, 512000: 180}
    layerset = {18: 2000000,  32: 1600000, 128: 1000000, 512: 325000, 5024: 12000, 100048: 610, 512000: 45}
    layersetavx = {18: 2000000,  32: 1600000, 128: 1000000, 512: 200000, 5024: 7250, 100048: 150, 512000: 15}

    with open("matmul_results.csv", "a") as file:
        file.write("threads,skew,src_mem_kb,total_mem_kb,execution_time,inputdatacached,NoAMX\n")
        for cacheinput in ("0"):
            for src_mem_kb in src_mem_sizes_kb:
                layer = layersetavx[src_mem_kb] if args.noamx else layerset[src_mem_kb]
                for threads in ompthreads_values:
                    for skew in skew_values:
                        src_row, src_col, weight_row, weight_col, total_mem_mb = calculate_dimensions(src_mem_kb, skew, batch_size)
                        status = os.system(command)

                        # Check if the command was executed successfully
                        if status == 0:
                            print('Ram-cache and Swap Cleared')
                        else:
                            print('An error occurred')
                        execution_time = run_matmul(src_row, src_col, weight_row, weight_col, batch_size, threads, layer, total_mem_mb, cacheinput, args)
                        time.sleep(2)
                        print("threads,skew,src_mem_kb,total_mem_mb,execution_time,cacheinput\n")
                        print(f"{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{cacheinput}\n")
                        file.write(f"{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{cacheinput},{str(args.noamx).lower()}\n")


if __name__ == "__main__":
    main()

