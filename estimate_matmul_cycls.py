# Constants
mat_size = 96
operations_per_matmul = mat_size ** 3
num_matmuls = 80
operations_per_cycle_compute = 512  # Updated: AVX-512 with FMA for bf16
operations_per_cycle_memory = 24  # Updated: 48 bytes / 2 bytes per bf16

# Calculate total operations
total_operations = operations_per_matmul * num_matmuls

# Calculate cycles needed for all operations under ideal conditions
ideal_cycles_compute = total_operations / operations_per_cycle_compute
ideal_cycles_memory = total_operations / operations_per_cycle_memory

# The actual cycles are constrained by the slower of computation and memory bandwidth
actual_cycles = max(ideal_cycles_compute, ideal_cycles_memory)

# Estimate effective cycles considering memory operations and cache effects
average_additional_cycles_per_access = 100e-9 / (1 / 2.5e9)  # 2.5 GHz assumed for cycle time calculation
effective_cycles = actual_cycles * (1 + average_additional_cycles_per_access)

print(f"Total Operations: {total_operations}")
print(f"Ideal Cycles for Computation: {ideal_cycles_compute}")
print(f"Ideal Cycles for Memory: {ideal_cycles_memory}")
print(f"Actual Cycles (considering slower of compute or memory): {actual_cycles}")
print(f"Effective Cycles (considering memory latency): {effective_cycles}")
print(f"Effective time in S : {effective_cycles * (1 / 2.5e9) }")

L1_hit_prob = 0.2
L1_miss_L2_hit_prob = 0.08
L1_miss_L2_miss_prob = 0.72

L1_latency = 10  # ns
L2_latency = 100  # ns
memory_latency = 300  # ns

effective_latency = (
    L1_hit_prob * L1_latency +
    L1_miss_L2_hit_prob * (L1_latency + L2_latency) +
    L1_miss_L2_miss_prob * (L1_latency + L2_latency + memory_latency)
)

print(f"Effective Average Latency per Access: {effective_latency} ns")

