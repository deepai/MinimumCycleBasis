#Ist argument implies the Input filename.
#2nd argument implies the output filename.
#3rd argument implies the number of threads.

mcb/mcb_cycle_deg_two_remove $1 $2"_R.txt" $3
mcb/mcb_cycle $1 $2"_W.txt" $3
mcb/mcb_non_cpu_rd $1 $2"_C.txt" $3
