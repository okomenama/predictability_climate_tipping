#!/bin/bash
set -x

export EXAMAZONOUTDIR='../data/rev_amazon2'
export EXAMAZONFIGDIR='../output/rev_amazon2'
export EXNUM='experiment1'

mkdir -p ${EXAMAZONFIGDIR}
mkdir -p ${EXAMAZONOUTDIR}
mkdir -p ${EXAMAZONOUTDIR}/experiment1
mkdir -p ${EXAMAZONFIGDIR}/experiment1

echo "Output is =${EXAMAZONOUTDIR}"
export EXDATADIR=${EXAMAZONOUTDIR}
export EXOUTDIR=${EXAMAZONFIGDIR}

for i in `seq -0 4`
do
echo "simplified TRIFFID"
export AMPLITUDE=${i}
echo "${i} - amplitude = ${AMPLITUDE:${i}}"
python3 ./amazon_tip_num_obs.py
python3 ./amazon_tip_num_non_obs.py
done
python3 ./figure2.py