#!/bin/bash
###############################################################################

GLOBAL_PATH='/faculty/jhou4/Projects/QA_protein/tools/DNSS2/';

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <targetid> <alignment> <outdir>"; 
  exit;
fi

targetid=$1
aln_in=$2
output=$3

cd $GLOBAL_PATH

mkdir -p $output

source /faculty/jhou4/tools/TripletRes_post_CASP13/TripletRes_venv_gpu1/bin/activate

python /faculty/jhou4/tools/TripletRes_post_CASP13/TripletRes.py $aln_in $output/$targetid.rr



