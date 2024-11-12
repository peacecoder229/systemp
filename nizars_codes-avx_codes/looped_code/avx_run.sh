echo "loadin emon driver"
/opt/intel/sep/sepdk/src/insmod-sep
echo "starting emon data collection"
/pnpdata/clucene_benchmark_new/emon/run_emon.sh

echo "runnig now the avx workload"
./cpu_stress_avx
source /pnpdata/clucene_benchmark_new/emon/sep_vars.sh
emon -stop
echo "Emon Collection Stopped"
