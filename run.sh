#Ist argument implies the Input filename.
#2nd argument implies the number of threads.

mcb/mcb_cycle $1 . $2 >file1.txt
mcb/mcb_cycle_deg_two_remove $1 . $2 >file2.txt
