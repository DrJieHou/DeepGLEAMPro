#!/bin/bash
###############################################################################

GLOBAL_PATH='/faculty/jhou4/Projects/QA_protein/tools/DNSS2/';

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <seq.fasta> <alignment> <outdir>"; 
  exit;
fi

FASTA=$1
aln_in=$2
output=$3

cd $GLOBAL_PATH

mkdir -p $output
source /faculty/jhou4/Projects/QA_protein/tools/DNSS2/dnss_venv/bin/activate

perl run_DNSS2_aln.pl  -seq $FASTA -aln $aln_in -file -out $output



