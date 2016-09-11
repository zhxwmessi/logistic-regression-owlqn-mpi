#!/bin/bash

#tt=`date`
#mkdir backup/"$tt"
#mv train backup/"$tt"
#mv *.log backup/"$tt"
#mv core backup/"$tt"
#make
#rm log/*
process_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
ssh worker@$ip rm /home/worker/xiaoshu/DML/logistic_regression_mpi/train
done
scp train worker@10.101.2.89:/home/worker/xiaoshu/DML/logistic_regression_mpi/.
scp train worker@10.101.2.90:/home/worker/xiaoshu/DML/logistic_regression_mpi/.
mpirun -f ../hosts -np $process_number  ./train owlqn 5 10 ./data/agaricus.txt.train ./data/agaricus.txt.test
