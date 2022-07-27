#!/bin/bash
###############################################################################

GLOBAL_PATH='/faculty/jhou4/Projects/QA_protein/tools/DNSS2/';

if [[ $# -ne 2 ]]; then
  echo "Usage: $0  <directory> <outdir>"; 
  exit;
fi

modeldir=$1
output=$2

cd $GLOBAL_PATH

mkdir -p $output

source /lab/hou/tools/trRosetta/trRosetta_virenv/bin/activate
python /faculty/jhou4/tools/QDeep/scripts/ros_energy_v2.py -d $modeldir -o $output 




