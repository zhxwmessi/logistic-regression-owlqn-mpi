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
ssh worker@$ip rm /home/worker/xiaoshu/logistic-regression-owlqn-mpi/train
done
scp train worker@10.101.2.89:/home/worker/xiaoshu/logistic-regression-owlqn-mpi/.
scp train worker@10.101.2.90:/home/worker/xiaoshu/logistic-regression-owlqn-mpi/.
#mpirun -f ./hosts -np $process_number  ./train owlqn 10 1000 ./data/train ./data/test
#mpirun -f ./hosts -np $process_number  ./train owlqn 500 1000 ./data/agaricus.txt.train ./data/agaricus.txt.test
mpirun -f ./hosts -np $process_number  ./train owlqn 200 300000 ./data/v2v_train ./data/v2v_test
