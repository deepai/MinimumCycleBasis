#$1 is the input filename
#$2 is the number of nodes in the input file.
#$3 is the outputfile.
#$4 is the number of threads for computing mcb.

ulimit -s unlimited

rm bicc_output/* -f

echo "Invoking bicc_decomposition..."

bicc/bicc_decomposition $1 bicc_output/ 0 $2 0 1 >> /dev/null

statsFile="bicc_output/stats"

while read line;
do
	#next steps
 	#echo $firstline ; file which is used to call the maxclique pmc
 	
 	filename=$(echo $line|cut -f 1 -d " ")".mtx"

 	bicc/Relabeller "bicc_output/"$filename $filename 1

 	sh run.sh $filename "Results/"$3 $4

 	rm -rf $filename

done < $statsFile