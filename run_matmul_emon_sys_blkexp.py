import subprocess
import argparse
import math
import time
import os

def setcache(lpcore, hpcore, lpcache, hpcache):
    cmdrst = "pqos -R"
    cmdclosdef = f"pqos -a llc:7={lpcore}"
    cmdsetl3 = f"pqos -e llc:7={lpcache}"
    showclos = "pqos -s"
    os.system(cmdrst)
    print(cmdclosdef)
    print(cmdsetl3)
    os.system(cmdclosdef)
    os.system(cmdsetl3)
    #os.system(showclos)

def setl2cat(lpcore, hpcore, lpcache, hpcache):
    cmdrst = "pqos -R"
    cmdclosdef = f"pqos -a llc:7={lpcore}"
    cmdsetl3 = f"pqos -e l2:7={lpcache}"
    showclos = "pqos -s"
    os.system(cmdrst)
    print(cmdclosdef)
    print(cmdsetl3)
    os.system(cmdclosdef)
    os.system(cmdsetl3)

def calculate_dimensions(src_mem_kb, skew, bs):
    #src_col = math.sqrt(src_mem_kb * 1024 / 2 / skew / bs) 
    src_col = math.sqrt(src_mem_kb * 1024 / 2 / skew) 
    src_row = src_col * skew
    weight_row = src_col
    weight_col = weight_row * skew
    #total_mem_mb = (src_col * src_row + weight_row * weight_col + src_row * weight_col) * 4 * bs / 1024 / 1024
    total_mem_mb = (src_col * src_row + weight_row * weight_col + src_row * weight_col) * 2 * bs / 1024 / 1024 # bf16 2 bytes
    return int(src_row), int(src_col), int(weight_row), int(weight_col), total_mem_mb

def run_matmul(src_row, src_col, weight_row, weight_col, batch_size, threads, layer, total_mem_mb, cacheinput, args ):
    core = 1 if (threads == 1) else (threads - 1)
    startcore = 1 if (threads == 1) else 0

    core2 = 65 if (threads == 1) else (threads + 63)
    startcore2 = 65 if (threads == 1) else 64
    usebenchdnn = args.usebenchdnn
    #cmd = f"numactl -C 0-{core} ./matmul_args --src_row {src_row} --src_col {src_col} " \
    #cmd = f"numactl -C 0-{core} ./matmul_bf16 --src_row {src_row} --src_col {src_col} " \
    env = ""

    if args.noamx:
        env += "ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 "
    if args.dnnverbose:
        env += "ONEDNN_VERBOSE=1 "
    if args.matmulverbose:
        env += "MATMUL_VERBOSE=1"
     #Add the additional environment variables
    if args.blk:
    #update to chnage N_BLK to 64 minimum and modify N_CHUNK_SZ
        if args.blk >= 64:
            n_blk = 64
            n_sz = args.blk // 64
        else:
            n_blk = args.blk
        env += f" M_CHUNK_SZ=1 N_CHUNK_SZ={n_sz} BATCH_SZ={batch_size} "
        #env += f"M_BLK={args.blk} N_BLK={args.blk} K_BLK={args.blk} "
        env += f"M_BLK={args.blk} N_BLK={n_blk} "
    else:
        pass

# Strip any trailing whitespace
    env = env.strip()

    print(f"Environment string: '{env}'")

    #env = "ONEDNN_VERBOSE=1 ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 "
    cmd = f"{env} numactl -C {startcore}-{core} ./matmul_sereilbf16 --src_row {src_row} --src_col {src_col} " \
          f"--weight_row {weight_row} --weight_col {weight_col} " \
          f"--batch_size {batch_size} --ompthreads {threads} --layer {layer} --cachedata {cacheinput} &" \
          f" {env} numactl -C {startcore2}-{core2} ./matmul_sereilbf16 --src_row {src_row} --src_col {src_col} " \
          f"--weight_row {weight_row} --weight_col {weight_col} " \
          f"--batch_size {batch_size} --ompthreads {threads} --layer {layer} --cachedata {cacheinput}"
    
    benchdnncmd = f"{env} OMP_NUM_THREADS={threads} numactl -C {startcore}-{core} /root/oneDNN/build/tests/benchdnn/benchdnn --mode=P --matmul --dt=bf16:bf16:bf16 " \
          f"--stag=abc --wtag=abc --dtag=abc {batch_size}x{src_row}x{src_col}:1x{weight_row}x{weight_col} & " \
          f"{env} numactl -C {startcore2}-{core2} /root/oneDNN/build/tests/benchdnn/benchdnn --mode=P --matmul --dt=bf16:bf16:bf16 " \
          f"--stag=abc --wtag=abc --dtag=abc {batch_size}x{src_row}x{src_col}:1x{weight_row}x{weight_col}"
          #f"--stag=abc --wtag=abc --dtag=abc {batch_size}x{src_row}x{src_col}:{batch_size}x{weight_row}x{weight_col}"

    #print(cmd)
    mem_mb = round(total_mem_mb, 1)
    mem_kb = round(total_mem_mb*1000, 1)
    time.sleep(1)
    #new_cmd = f'tmc -T all -x rdas -i "NA" -Z metrics2 -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "thread{threads}_{mem_mb}MB_SRCrows{src_row}" -n -u -c "{cmd}"'
    #new_cmd = f'tmc -T all -x rdas -i "NA" -Z metrics2 -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "{mem_kb}kB_NoAMX{str(args.noamx).lower()}" -n -u -c "{cmd}"'
    #new_cmd = f'tmc -T all -x rdas -i "NA" -Z metrics -e /opt/intel/sep/config/edp/filtered_memcpularge.txt -D "{mem_kb}kB_NoAMX{str(args.noamx).lower()}" -n -u -c "{cmd}"'
    new_cmd = f'tmc -T all -x rdas -i "NA" -Z flex -e /opt/intel/sep/config/edp/filtered_memcpularge_mem.txt -D "2Sreduced_{mem_kb}kB_NoAMX{str(args.noamx).lower()}" -n -u -c "{cmd}"'
    #print(new_cmd) 
    #finalcmd = new_cmd if args.emon elif usebenchdnn benchdnncmd else cmd  
    finalcmd = new_cmd if args.emon else (benchdnncmd if usebenchdnn else cmd)

    print(finalcmd) 
    #result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    #result = subprocess.run(new_cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    result = subprocess.run(finalcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    time.sleep(5)
    output = result.stdout
    #time_str = output.split("Elapsed time: ")[1].split(" seconds")[0]
    #return float(time_str)
    print(output)
    import re

    # Assuming result.stdout contains the output from your subprocess command

    # Use a regular expression to find and extract the elapsed time
    #match = re.search(r'Elapsed time: ([\d.]+) seconds', output)
    if usebenchdnn:
        matches = re.findall(r'total perf: min\(ms\):[\d.]+ avg\(ms\):([\d.]+)', output)
    else:
        matches = re.findall(r'matmultime=([\d.]+)seconds', output)
    if matches:
        #time_str = match.group(1)
        time_str = matches[0]# This will be the string '0.645657' if the line is found
        socket1 = matches[1]
        print(f"Elapsed time: {time_str} seconds")
        print(f"All matches {matches}")
    else:
        print("Elapsed time not found in the output.")
    m_values = re.search(r'bgmmc: M =\s+(\d+), M_blk =\s+(\d+), M_tail =\s+(\d+), M_chunks =\s+(\d+), M_chunk_elems =\s+(\d+), num_M_blocks =\s+(\d+), M_chunk_size =\s+(\d+)', output)
    n_values = re.search(r'bgmmc: N =\s+(\d+), N_blk =\s+(\d+), N_tail =\s+(\d+), N_chunks =\s+(\d+), N_chunk_elems =\s+(\d+), num_N_blocks =\s+(\d+), N_chunk_size =\s+(\d+)', output)
    k_values = re.search(r'bgmmc: K =\s+(\d+), K_blk =\s+(\d+), K_tail =\s+(\d+), K_chunks =\s+(\d+), K_chunk_elems =\s+(\d+)', output)
    combined_values = ""
    if m_values:
        M, M_blk, M_tail, M_chunks, M_chunk_elems, num_M_blocks, M_chunk_size = m_values.groups()
        combined_values += f"M values: M={M}, M_blk={M_blk}, M_tail={M_tail}, M_chunks={M_chunks}, M_chunk_elems={M_chunk_elems}, num_M_blocks={num_M_blocks}, M_chunk_size={M_chunk_size},"

    if n_values:
        N, N_blk, N_tail, N_chunks, N_chunk_elems, num_N_blocks, N_chunk_size = n_values.groups()
        combined_values += f"N values: N={N}, N_blk={N_blk}, N_tail={N_tail}, N_chunks={N_chunks}, N_chunk_elems={N_chunk_elems}, num_N_blocks={num_N_blocks}, N_chunk_size={N_chunk_size},"

    if k_values:
        K, K_blk, K_tail, K_chunks, K_chunk_elems = k_values.groups()
        combined_values += f"K values: K={K}, K_blk={K_blk}, K_tail={K_tail}, K_chunks={K_chunks}, K_chunk_elems={K_chunk_elems},"


    return float(time_str), float(socket1), combined_values.strip()

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
    parser.add_argument('--usebenchdnn', action='store_true', help="Indicate whether to use benchdnn test")
    parser.add_argument('--noamx', action='store_true', help="Enable emon monitoring")
    parser.add_argument('--dnnverbose', action='store_true', help="Enable emon monitoring")
    parser.add_argument('--matmulverbose', action='store_true', help="Enable emon monitoring")
    parser.add_argument('--blk', type=int, help="provide M,N and K blocksize like 64, 512 etc", default=None)
    args = parser.parse_args()
    emon = args.emon
    #src_mem_sizes_kb = [18, 32, 128, 512, 1152]  # in KB
    #src_mem_sizes_kb = [18, 32, 128]  # in KB
    #src_mem_sizes_kb = [512, 1152]  # in KB
    #src_mem_sizes_kb = [114688, 32768, 8192]  # in KB
    #src_mem_sizes_kb = [2048, 32768, 114688, 256000]  # in KB
    #src_mem_sizes_kb = [114688]  # in K B 7662 rows an cols
    src_mem_sizes_kb = [ 2048, 32768, 98304 ]  # in KB 1024 rows and cols => 2048   4096 rows adn cols is 32768
    #src_mem_sizes_kb = [18, 32, 128, 512, 5024, 100048, 512000]
    #src_mem_sizes_kb = [256, 16384, 100048, 512000]
    #src_mem_sizes_kb = [512, 1024, 2048, 3048, 6144]
    ompthreads_values =[64, 32]
    #ompthreads_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    skew_values = [1]
    batch_size = 32
    #layer = 40
    #layerset = {18: 2000000,  32: 1600000, 128: 1000000, 512: 1000000, 5024: 40000, 100048: 16800, 512000: 180}
    #layerset = {18: 2000000,  32: 1600000, 128: 1000000, 512: 325000, 256: 200000, 5024: 12000, 16384:4000, 100048: 610, 512000: 45}
    #layersetavx = {18: 2000000,  32: 1600000, 128: 1000000, 512: 200000, 256: 100000, 5024: 7250, 16384:2025, 100048: 150, 512000: 15}
    layerset = {18: 2000000,  32: 1600000, 128: 1000000, 512: 325000, 256: 1700000, 5024: 12000, 16384:16000, 100048: 1810, 512000: 110}
    layersetavx = {18: 2000000,  32: 1600000, 128: 1000000, 512: 200000, 256: 1000000, 5024: 7250, 16384:4050, 100048: 300, 512000: 25}

    with open("matmul_multicore_final.csv", "a") as file:
        #file.write("threads,skew,src_mem_kb,total_mem_kb,execution_timeS0,execution_timeS1,inputdatacached,NoAMX\n")
        file.write(f"BLK,batch_size,src_row,src_col,weight_row,weight_col,threads,skew,src_mem_kb,total_mem_mb,execution_time,S1,cacheinput,noamx,blockinfo\n")
        for cacheinput in ("0"):
            for src_mem_kb in src_mem_sizes_kb:
                #layer = layersetavx[src_mem_kb] if args.noamx else layerset[src_mem_kb]
                layer = 100  
                for threads in ompthreads_values:
                    for skew in skew_values:
                        src_row, src_col, weight_row, weight_col, total_mem_mb = calculate_dimensions(src_mem_kb, skew, batch_size)
                        print(f"{src_col},{src_mem_kb}")
                        #status = os.system(command)

                        # Check if the command was executed successfully
                        if status == 0:
                            print('Ram-cache and Swap Cleared')
                        else:
                            print('An error occurred')

                        #for cache in [ "0xffff" ]:
                            #setcache("0-63", None, cache, None) 
                        #setl2cat("0-63", None, cache, None) 
                        execution_time, S1, blockinfo = run_matmul(src_row, src_col, weight_row, weight_col, batch_size, threads, layer, total_mem_mb, cacheinput, args)
                        time.sleep(2)
                        print("threads,skew,src_mem_kb,total_mem_mb,execution_times0,execution_times1,cacheinput\n")
                        print(f"{str(args.blk)},{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{cacheinput}\n")
                        file.write(f"{str(args.blk)},{batch_size},{src_row},{src_col},{weight_row},{weight_col},{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{S1},{cacheinput},{str(args.noamx).lower()},{blockinfo}\n")
                        #file.write(f"{src_row},{src_col},{weight_row},{weight_col},{threads},{skew},{src_mem_kb},{total_mem_mb},{execution_time},{S1},{cacheinput},{str(args.noamx).lower()},{cache}\n")


if __name__ == "__main__":
    main()

