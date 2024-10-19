#!/bin/bash
set -x

export NAME='AMOC'
export EXAMOCOUTDIR='../data/rev_AMOC'
export EXAMOCFIGDIR='../output/rev_AMOC'
export EXNUM='experiment_acc'

mkdir -p ${EXAMOCFIGDIR}
mkdir -p ${EXAMOCOUTDIR}
mkdir -p ${EXAMOCFIGDIR}/experiment_acc
mkdir -p ${EXAMOCOUTDIR}/experiment_acc

echo "Output is =${EXAMOCOUTDIR}"
export EXDATADIR=${EXAMOCOUTDIR}
export EXOUTDIR=${EXAMOCFIGDIR}
for i in `seq -0 4`
do
echo "AMOC_two_box"
export AMPLITUDE=${i}
echo "${i} - amplitude = ${AMPLITUDE:${i}}"
python3 ./AMOC_tip_num_obs_acc.py
done

python3 ./figure3.py