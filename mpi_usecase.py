from mpi4py import MPI
import numpy as np
import sys

# Initialize the MPI environment
comm = MPI.COMM_WORLD
# Get the number of processes
size = comm.Get_size()
print("MPI Size found is %s\n" %(size))
# Get the rank of the process
rank = comm.Get_rank()

# Function to split data among processes
def split_data(data, size):
    # Split data into chunks for each process
    chunks = np.array_split(data, size)
    return chunks

# Example data array
num = int(sys.argv[1])
data = np.arange(num)  # An array of integers from 0 to 99

# Split data only on the root process
if rank == 0:
    chunks = split_data(data, size)
else:
    chunks = None

# Scatter the chunks to all processes
chunk = comm.scatter(chunks, root=0)

# Each process computes its partial sum
partial_sum = np.sum(chunk)

# Gather all partial sums back to the root process
total_sum = comm.reduce(partial_sum, op=MPI.SUM, root=0)

# Only the root process prints the result
if rank == 0:
    print(f"Total sum: {total_sum}")


