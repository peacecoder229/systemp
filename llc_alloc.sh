#!/bin/bash
pqos -R
pqos -a llc:7=24-35,60-71
pqos -e "llc:7=0x3"
pqos -a llc:3=0-23,36-59
pqos -e "llc:3=0xffffc"

