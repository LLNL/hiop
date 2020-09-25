#!/usr/bin/awk -f
function abs(v) {return v < 0 ? -v : v}
BEGIN {
  columns[0]="iter"
  columns[1]="objective"
  columns[2]="inf_pr"
  columns[3]="inf_du"
  columns[4]="lg(mu)"
  columns[5]="alpha_du"
  columns[6]="alpha_pr"
  columns[7]="linesrch"
  fail=0
  tol=0.00001
}
{
  len=split($0, data, " ")
  if(len!=15) {
    printf "Got incorrect number of fields (%d)\n", len
    exit(1)
  }
  for(i=2;i<8;i++) {
    j=i + 7
    if(abs((data[i] + 0)-(data[j] + 0)) > tol) {
      printf "Found difference on iteration %d in field %s:\n", data[0], columns[i]
      printf "1: %-20f 2: %-20f\n", data[i], data[j]
      fail++
    }
  }
  if(data[8]!=data[15]) {
    printf "Found difference on iteration %d in field %s:\n", data[0], columns[7]
    printf "1: %-20s 2: %-20s\n", data[8], data[15]
    fail++
  }
}
END {
  if (fail) {
    print "Found failures."
    exit(1)
  }
  print "Example outputs matched!"
}
