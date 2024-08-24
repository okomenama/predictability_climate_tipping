#!/bin/bash
set -x

export EXAMOCOUTDIR='../data/rev_AMOC'
export EXAMOCFIGDIR='../output/rev_AMOC'
export EXNUM='experiment2'

mkdir -p ${EXAMOCFIGDIR}
mkdir -p ${EXAMOCOUTDIR}
mkdir -p ${EXAMOCFIGDIR}/experiment2
mkdir -p ${EXAMOCOUTDIR}/experiment2

echo "Output is =${EXAMOCOUTDIR}"
export EXDATADIR=${EXAMOCOUTDIR}
export EXOUTDIR=${EXAMOCFIGDIR}
for i in `seq -0 4`
do
echo "AMOC_two_box"
export AMPLITUDE=${i}
echo "${i} - amplitude = ${AMPLITUDE:${i}}"
python3 ./AMOC_tip_num_obs.py
done

python3 ./figure3.py