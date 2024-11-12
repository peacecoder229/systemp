#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the flippingBits function below.
def flippingBits(n):
    return ~n

if __name__ == '__main__':

    q = int(input())

    for q_itr in range(q):
        n = int(input())

        result = flippingBits(n)
        print(str(result) + "\n")
    
