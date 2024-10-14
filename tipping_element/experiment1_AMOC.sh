#!/bin/bash
set -x

export NAME='AMOC'
export EXAMOCOUTDIR='../data/rev_AMOC'
export EXAMOCFIGDIR='../output/rev_AMOC'
export EXNUM='experiment1'

mkdir -p ${EXAMOCFIGDIR}
mkdir -p ${EXAMOCOUTDIR}
mkdir -p ${EXAMOCFIGDIR}/experiment1
mkdir -p ${EXAMOCOUTDIR}/experiment1

echo "Output is =${EXAMOCOUTDIR}"
export EXDATADIR=${EXAMOCOUTDIR}
export EXOUTDIR=${EXAMOCFIGDIR}

for i in `seq -0 4`
do
echo "AMOC two box"
export AMPLITUDE=${i}
echo "${i} - amplitude = ${AMPLITUDE:${i}}"
python3 ./AMOC_tip_num_obs.py
python3 ./AMOC_tip_num_non_obs.py
done
python3 ./figure2.py