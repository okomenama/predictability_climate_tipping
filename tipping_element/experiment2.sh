#!/bin/bash
set -x

export NAME='AMAZON'
export EXAMAZONOUTDIR='../data/rev_amazon'
export EXAMAZONFIGDIR='../output/rev_amazon'
export EXNUM='experiment2'
export REDCHAFRE=0

mkdir -p ${EXAMAZONFIGDIR}
mkdir -p ${EXAMAZONOUTDIR}
mkdir -p ${EXAMAZONFIGDIR}/experiment2
mkdir -p ${EXAMAZONOUTDIR}/experiment2

echo "Output is =${EXAMAZONOUTDIR}"
export EXDATADIR=${EXAMAZONOUTDIR}
export EXOUTDIR=${EXAMAZONFIGDIR}
for i in `seq -0 4`
do
echo "simplified TRIFFID"
export AMPLITUDE=${i}
echo "${i} - amplitude = ${AMPLITUDE:${i}}"
python3 ./amazon_tip_num_obs.py
done

python3 ./figure3.py
