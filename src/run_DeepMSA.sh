#!/bin/bash -l

if [ $# -ne 3 ]
then
        echo "need three parameters: target id, input fasta file, output directory."
        exit 1
fi

targetid=$1
fastafile=$2
outputdir=$3

mkdir -p $outputdir
cd $outputdir


python /faculty/jhou4/tools/trRosetta/DeepMSA/hhsuite2/scripts/build_MSA.py $fastafile -hhblitsdb=/faculty/jhou4/Projects/bio_databases/uniclust30_2017_10/uniclust30_2017_10  -hmmsearchdb=/faculty/jhou4/Projects/bio_databases/metaclust_2018_06/metaclust_50.fasta
