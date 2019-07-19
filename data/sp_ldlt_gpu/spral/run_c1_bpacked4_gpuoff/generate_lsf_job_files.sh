#/bin/sh

input="list_matrices.txt"
template="template_job.lsf"
currdir=`pwd`

sed 's/xxxname/spral_ssids/g' template_job.lsf > job.lsf

echo "mkdir logs" >> job.lsf

while IFS= read -r line
do
  filename=$(basename $line)

  echo "ln $line matrix.rb" >> job.lsf

  jscmd="jsrun -n1 -a1 -c4 -bpacked:4 -g1  ./spral_ssids  > logs/${filename}.log"
  echo $jscmd >> job.lsf

  echo "unlink matrix.rb" >> job.lsf
  echo "echo 'Done with ${filename}'" >> job.lsf
  echo "" >> job.lsf
done < $input
