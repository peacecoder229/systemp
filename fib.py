#!/usr/bin/env python3
import datetime

def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)

def fibopt(n, fiblist):
    #print(str(n))
    if n <=1:
        return n
    elif fiblist[n]:
        return fiblist[n]
    else:
        fiblist[n] = fibopt(n-1, fiblist) + fibopt(n-2, fiblist)
        return fiblist[n]
        
if __name__ == "__main__":
        
    n = int(input("Enter num:"))
    print("number is %d\n" %n)
    t1 = datetime.datetime.now()

    fiblist = list()
    fiblist = (n + 1) * [0]
    #print(len(fiblist))

    #print("fib for 20 is %d\n" %fib(n))
    print("fib for num %d is %d\n" %(n, fibopt(40, fiblist)))

    t2 = datetime.datetime.now()
    delta = t2-t1

    print("time taken = %d" %delta.microseconds)
