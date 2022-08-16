# Deep graph-based learning for estimating the accuracy of protein's model structure

Updating


# Installation

Clone the repository and install the dependencies listed above.

# Requirements

(Optional) Install the virtual environment for module installation:

```
python3 -m venv trRosetta
source trRosetta/trRosetta_virenv/bin/activate
```

The prediction pipeline uses Python3 and requires the following modules:

```
pip install --upgrade pip
pip install keras==2.1.6
pip install numpy==1.15.2
pip install matplotlib
pip install scipy
pip install numba
pip install sklearn
pip install --upgrade h5py
pip install tensorflow==1.13.1
pip install biopython
pip install tensorflow
```


# Installation

# Dependency

## library for rosetta


## make sure the python version matches the rosetta file
pip install ./pyrosetta-2022.29+release.fc0fea1-cp39-cp39-linux_x86_64.whl


cd ./test

sh ./src/run_DeepMSA.sh 1k4mA ./1k4mA.fasta ./test/msa &> ./runDeepMSA.log  & 



### generate DNSS 
sh ./src/run_DNSS2_aln.sh ./test/3f52A.fasta ./test/3f52A.aln ./test/3f52A

perl ./src/format_dnss.pl  ./test/3f52A/3f52A.dnss 3f52A ./test/3f52A/3f52A.ss_DNSS



### generate Contact
sh ./src/run_TripletRes_aln.sh 3f52A ./test/3f52A.aln ./test/3f52A_rr



# Generage rosetta energy scores 

sh ./src/run_rosetta_scores.sh ./test/3d_models/3f52A ./test/rosetta &> ./test/rosetta.log & 


# Generate dssp 


perl ./src/P5_pdb2dssp_batch.pl ./src/dssp  ./test/3d_models/3f52A  ./test/dssp

perl ./src/P6_dssp2ssa_batch.pl  ./test/dssp  ./test/3f52A.fasta  ./src/dssp2dataset_dnss.pl


# 15



python ./src/generate_feature.py -t '3f52A' -f ./test/3f52A.fasta -s ./test/3f52A/3f52A.ss_DNSS -r ./test/3f52A_rr/3f52A.rr  -p ./test/3f52A/temp/pssm/3f52A.pssm  -d ./test/3d_models/3f52A  -k  ./test/dssp  -e ./test/rosetta  -o ./test/QA_features/3f52A.tsv -w 15


### make prediction 

python ./src/predict_score.py -w ./test/  -f ./test/QA_features/3f52A.tsv -m ./src/optimal_models_withaln


