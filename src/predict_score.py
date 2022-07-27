#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pickle
import argparse
from collections import defaultdict
import datetime 
import os 
import typing
from typing import Any, Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

### Generate pickle file 
import re 
import sys 
import os
from os import listdir
from os.path import isfile, join
import collections

import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np

import pickle
import numpy as np
import time 

import operator
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



from scipy.spatial.distance import pdist, squareform
import random 



def shape_list(x):
    """Return list of dimensions of a tensor, statically where possible.

    Like `x.shape.as_list()` but with tensors instead of `None`s.

    Args:
    x: A tensor.
    Returns:
    A list with length equal to the rank of the tensor. The n-th element of the
    list is an integer when that dimension is statically known otherwise it is
    the n-th element of `tf.shape(x)`.
    """
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
    
class ProteinQAData_Generator(tf.keras.utils.Sequence):
    def __init__(self, data_path, featurefile,  min_seq_size= 0,  max_seq_size= 10000, batch_size = 5, max_msa_seq = 128, max_id_nums = 1000000):
        
        self.data_path = data_path
        
        self.featurefile = featurefile
        self.max_id_nums = max_id_nums
        self.min_seq_size = min_seq_size
        self.max_seq_size = max_seq_size
        self.batch_size = batch_size
        self.max_msa_seq = max_msa_seq
        
        self.path_msa = os.path.join(self.data_path, "msa/")
        self.path_label = os.path.join(self.data_path, "dssp/")
        self.path_QAfeature = os.path.join(self.data_path, "QA_features/")
        
        
        self.id2seq = self.get_filenames(head = self.max_id_nums)
        self.id_list = list(self.id2seq.keys()) # '3gykA|iter0.2.pdb.pdb'
        
        ### speed up running using batch list by length
        batchsize = 10
        self.seq2id = defaultdict(list)
        aa = 0
        for seq_id, seq_len in sorted(self.id2seq.items()):
            self.seq2id[seq_len].append(seq_id)
        
        protein_model_id_list = []
        for seq_len in self.seq2id.keys():
            if seq_len < 100:
                batchsize = 60
            elif seq_len < 200:
                batchsize = 40
            elif seq_len < 300:
                batchsize = 20
            elif seq_len < 400:
                batchsize = 10
            else:
                batchsize = 5
            
            if self.batch_size == 'single':
                batchsize = 1

            comb = self.seq2id[seq_len]
            batch_len = int(len(comb)/ batchsize)+1
            for index in range(batch_len):
                batch_list = comb[index * batchsize: (index + 1) * batchsize]
                protein_model_id_list.append(batch_list)
        
        self.protein_model_id_list = protein_model_id_list

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.protein_model_id_list))
        #self.indexes = np.arange(len(self.id_list))
        #np.random.shuffle(self.indexes)

    def __len__(self):
        #return int(len(self.id_list) / self.batch_size)
        return int(len(self.protein_model_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_id_list = self.protein_model_id_list[index]
        if len(batch_id_list) == 0:
            batch_id_list = random.sample(self.protein_model_id_list,1)[0]
            
        #batch_id_list = self.id_list[index * self.batch_size: (index + 1) * self.batch_size]
        target_feat_batch, msa_feat_batch, pdb_rosetta_batch, seq_pssm_batch, seq_ss_batch, seq_contact_batch, pdb_distance_pair_batch, model_aa_lddt_batch, model_gdt_batch  = self.collect_data(batch_id_list)

        return target_feat_batch, msa_feat_batch, pdb_rosetta_batch, seq_pssm_batch, seq_ss_batch, seq_contact_batch, pdb_distance_pair_batch, model_aa_lddt_batch
        #return target_feat_batch, msa_feat_batch,model_aa_lddt_batch


    def __call__(self):
        self.i = 0
        return self

    def get_filenames(self, head = 10000000):
    
        qadata = pd.read_csv(self.featurefile,sep='\t',header=0)
        files_no_ext = {} 
        for submodel in qadata['model'].unique():
            subset = qadata[qadata['model'] == submodel]
            targetid = subset['target'].unique()[0]
            seq_id = targetid + '|' + submodel
            seq_len = len(subset)
            files_no_ext[seq_id] = seq_len

            if len(files_no_ext) >= head:
                break
        
        return files_no_ext 
    
    def open_data(self, file_name):    
        with open(self.path_msa + file_name + '.aln') as train_f: 
            sample_msa = train_f.read().strip('\n').split('\n')
        return sample_msa
    
    def collect_data(self, batch_id_list):
        # ['3gykA|iter0.2.pdb.pdb']

        #target_feat_batch:  (None, L_max, 21)
        
        # (Part I): load sequence data
        # get maximum size from this batch 
        max_seq_len = 0
        for i in range(len(batch_id_list)):
            try:
                if self.id2seq[batch_id_list[i]] > max_seq_len:
                    max_seq_len = self.id2seq[batch_id_list[i]]
            except:
                print("batch_id_list[i]: ",batch_id_list[i])
                exit(-1)
                
        self.pad_size = max_seq_len
        target_feat_batch = np.full((len(batch_id_list), self.pad_size, 21), 0.0)
        target_dnss_batch = np.full((len(batch_id_list), self.pad_size, 73), 0.0)
        msa_feat_batch = np.full((len(batch_id_list), self.max_msa_seq, self.pad_size, 46), 0.0)
        
        ## qa features
        pdb_rosetta_batch = np.full((len(batch_id_list), self.pad_size,12),0.0)
        seq_pssm_batch = np.full((len(batch_id_list), self.pad_size,20),0.0)
        seq_ss_batch = np.full((len(batch_id_list), self.pad_size,15),0.0)
        seq_contact_batch= np.full((len(batch_id_list), self.pad_size, self.pad_size,1),0.0)
        pdb_distance_pair_batch= np.full((len(batch_id_list), self.pad_size, self.pad_size,1),0.0)
        model_aa_lddt_batch = np.full((len(batch_id_list), self.pad_size,1),0.0)
        model_gdt_batch = np.full((len(batch_id_list),1),0.0)


        ### here we can optimize file loading for the models from same target
        target2models = defaultdict(list)
        target2models_seq = defaultdict(list)
        for i in range(len(batch_id_list)):
            try:
                info = batch_id_list[i].split('|') # '3gykA|iter0.2.pdb.pdb'
            except:
                print("info = batch_id_list[i].split('|') batch_id_list[i]: ",batch_id_list[i])
                exit(-1)
            seq_len = self.id2seq[batch_id_list[i]]
            target_id = info[0] #'3gykA'
            target_model = info[1] #'iter0.2.pdb.pdb'
            target2models[target_id].append(batch_id_list[i])
            target2models_seq[target_id].append(seq_len)
            
        
        ### load models per target
        loaded_count = 0
        for target_id in target2models.keys():
            
            models = target2models[target_id] # ['3gykA|iter0.2.pdb.pdb','3gykA|iter0.2.pdb.pdb']
            
            
            seq_len_unique = np.unique(target2models_seq[target_id])
            
            
            if len(seq_len_unique)>1:
                print('Warning: pdb structures from same target have different length, check it')
                exit(-1)
            seq_len = seq_len_unique[0]

            ### (1) try load sequence pickle files
            sample_msa = self.open_data(target_id)   

            # (1) get msa encodings
            msas = sample_msa
            seq_orig = sample_msa[0]
            seq_len = len(seq_orig)
            msa_encoding = self.make_msa_features(sample_msa)

            # (2) get sequence feature
            aatype_1hot = self.sequence_to_onehot(seq_orig)
            target_feat = [aatype_1hot]  # Everyone gets the original sequence.]
            target_feat_encoding = tf.concat(target_feat, axis=-1)
            target_feat_encoding = tf.cast(target_feat_encoding, tf.float32).numpy() #(174, 21)

            msa_feat_encode,msa_extra_feat_encode = self.get_MSA_features(msa_encoding, max_msa_seq=self.max_msa_seq, max_extra_msa=5120)

            # (3) pad target feature into batch
            msa_feat_encode = tf.cast(msa_feat_encode, tf.float32).numpy() #msa_feat_encode:  (512, 174, 46)
            #msa_extra_feat_encode = tf.cast(msa_extra_feat_encode, tf.float32)
            
            
            
            feature = {}
            feature['target_feat'] = target_feat_encoding
            feature['msa_feat'] = msa_feat_encode

            
            ### (2) load model pickle files 
            qadata = pd.read_csv(self.path_QAfeature+'/'+target_id+'.tsv',sep='\t',header=0)
            target_pdb_feature = {}
            for submodel in qadata['model'].unique():
                subset = qadata[qadata['model'] == submodel]
                subst_rosetta, subst_pssm,subst_ss,model_distance,all_contact,subst_lddt, subst_gdt = self.get_feature_and_y_gcn(subset, contact_comb=contact_comb)
                if len(subst_rosetta) != seq_len:
                    print("The length of sequence doesn't match structure files")
                    exit(-1)
                pdb_feature = {}
                pdb_feature['pdb_rosetta'] = subst_rosetta
                pdb_feature['seq_pssm'] = subst_pssm
                pdb_feature['seq_ss'] = subst_ss
                pdb_feature['seq_contact_adjacent'] = all_contact
                pdb_feature['pdb_distance_pair'] = model_distance
                pdb_feature['model_aa_lddt'] = subst_lddt
                pdb_feature['model_gdt'] = subst_gdt
                
                target_pdb_feature[target_id+'|'+submodel] = pdb_feature
            
            import pickle 

            ## should load by batch
        

            seq_len = len(feature['target_feat'])
            
            #train_seq_core.append(target_feat_encoding) 
            #(None, 174, 21)
            target_feat_batch[loaded_count, 0:seq_len, :] = feature['target_feat']
            
            # (4) pad msa feature into batch
            #train_msa_core.append(msa_feat_encode)
            #msa_feat_encode:  (None, 512, 174, 46)
            msa_max_size = np.min([self.max_msa_seq, len(feature['msa_feat'])])
            msa_feat_batch[loaded_count, 0:msa_max_size, 0:seq_len, :] = feature['msa_feat'][0:msa_max_size]
                       
            #msa_extra_feat_encode:  (5120, 174, 23)
            #train_msa_extra_core.append(msa_extra_feat_encode)
            
        
            ##### (part 2) start loading each model of target the protein quality assessment data 
            
            for model in models:
                pdb_feature = target_pdb_feature[model]


                l = len(pdb_feature['model_aa_lddt'])

                if l != seq_len:
                    print("warning: pdb length is not equal to sequence length", l, "!=", seq_len)


                pdb_rosetta_batch[loaded_count, 0:l,:] = pdb_feature['pdb_rosetta']
                seq_pssm_batch[loaded_count, 0:l,:] = pdb_feature['seq_pssm']
                seq_ss_batch[loaded_count, 0:l,:] = pdb_feature['seq_ss']
                seq_contact_batch[loaded_count, 0:l, 0:l,0] = pdb_feature['seq_contact_adjacent']
                pdb_distance_pair_batch[loaded_count, 0:l, 0:l,0] = pdb_feature['pdb_distance_pair']
                model_aa_lddt_batch[loaded_count, 0:l,0] = pdb_feature['model_aa_lddt']
                model_gdt_batch[loaded_count] = pdb_feature['model_gdt']

                loaded_count += 1
                #label.append(temp_list)
        
        target_feat_batch = target_feat_batch[0:loaded_count]
        msa_feat_batch = msa_feat_batch[0:loaded_count]
        
        
        pdb_rosetta_batch=pdb_rosetta_batch[0:loaded_count] 
        seq_pssm_batch=seq_pssm_batch[0:loaded_count] 
        seq_ss_batch=seq_ss_batch[0:loaded_count] 
        seq_contact_batch=seq_contact_batch[0:loaded_count] 
        pdb_distance_pair_batch=pdb_distance_pair_batch[0:loaded_count] 
        model_aa_lddt_batch=model_aa_lddt_batch[0:loaded_count] 
        model_gdt_batch=model_gdt_batch[0:loaded_count] 
        
  
        return target_feat_batch, msa_feat_batch, pdb_rosetta_batch, seq_pssm_batch, seq_ss_batch, seq_contact_batch, pdb_distance_pair_batch, model_aa_lddt_batch, model_gdt_batch
    
    def onehot_to_sequence(self,onehot):
        restypes = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V'
        ]
        restypes_with_x = restypes + ['X']
        restype_order_with_x = {i:restype for i, restype in enumerate(restypes_with_x)}
        aa = ''
        for en in onehot:
            aa += restype_order_with_x[np.argmax(en)]
        
        return aa
    
    def onehot_to_msa(self,onehot):
        restypes = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V'
        ]
        restypes_with_x = restypes + ['X','-']
        restype_order_with_x = {i:restype for i, restype in enumerate(restypes_with_x)}
        aa = ''
        for en in onehot:
            aa += restype_order_with_x[np.argmax(en)]
        
        return aa

    def sequence_to_onehot(self, sequence):
        """Maps the given sequence into a one-hot encoded matrix."""
        map_unknown_to_x = True
        restypes = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V'
        ]
        restypes_with_x = restypes + ['X']
        restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

        mapping = restype_order_with_x

        num_entries = max(mapping.values()) + 1

        if sorted(set(mapping.values())) != list(range(num_entries)):
            raise ValueError('The mapping must have values from 0 to num_unique_aas-1 '
                         'without any gaps. Got: %s' % sorted(mapping.values()))

        one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

        for aa_index, aa_type in enumerate(sequence):
            if map_unknown_to_x:
                if aa_type.isalpha() and aa_type.isupper():
                    aa_id = mapping.get(aa_type, mapping['X'])
                else:
                    raise ValueError(f'Invalid character in the sequence: {aa_type}')
            else:
                aa_id = mapping[aa_type]
            one_hot_arr[aa_index, aa_id] = 1

        return one_hot_arr
    
    def make_msa_features(self,msas):
        """Constructs a feature dict of MSA features."""
        # The mapping here uses hhblits convention, so that B is mapped to D, J and O
        # are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
        # remaining 20 amino acids are kept in alphabetical order.
        # There are 2 non-amino acid codes, X (representing any amino acid) and
        # "-" representing a missing amino acid in an alignment.  The id for these
        # codes is put at the end (20 and 21) so that they can easily be ignored if
        # desired.
        HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,
                            'L': 9,'M': 10,'N': 11,'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,
                            'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,
                            }
        
        # Partial inversion of HHBLITS_AA_TO_ID.
        ID_TO_HHBLITS_AA = {
            0: 'A',
            1: 'C',  # Also U.
            2: 'D',  # Also B.
            3: 'E',  # Also Z.
            4: 'F',
            5: 'G',
            6: 'H',
            7: 'I',
            8: 'K',
            9: 'L',
            10: 'M',
            11: 'N',
            12: 'P',
            13: 'Q',
            14: 'R',
            15: 'S',
            16: 'T',
            17: 'V',
            18: 'W',
            19: 'Y',
            20: 'X',  # Includes J and O.
            21: '-',
        }
        
        
        msa_restypes = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V'
        ]
        restypes_with_x_and_gap = msa_restypes + ['X','-']


        if not msas:
            raise ValueError('At least one MSA must be provided.')

        int_msa = []
        uniprot_accession_ids = []
        species_ids = []
        seen_sequences = set()

        for sequence_index, sequence in enumerate(msas):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[HHBLITS_AA_TO_ID[res]]) for res in sequence])

        num_res = len(msas[0])
        num_alignments = len(int_msa)
        features_msa_encoding = np.array(int_msa, dtype=np.int32)
        return features_msa_encoding
    
    def get_MSA_features(self, msas_encoding, max_msa_seq=512, max_extra_msa=5120):
        """ (1) Sample MSA randomly, remaining sequences are stored as `extra_*`."""
        max_seq = max_msa_seq
        seq_msa = tf.constant(msas_encoding)
        num_seq = tf.shape(seq_msa)[0]
        shuffled = tf.compat.v1.random_shuffle(tf.range(1, num_seq))
        index_order = tf.concat([[0], shuffled], axis=0)
        num_sel = tf.minimum(max_seq, num_seq)
        sel_seq, not_sel_seq = tf.split(index_order, [num_sel, num_seq - num_sel])
        seq_msa_select = tf.gather(seq_msa, sel_seq) # (512, 174)
        seq_msa_select_extra = tf.gather(seq_msa, not_sel_seq) # (34819, 174)
        seq_msa_select_mask = tf.ones(shape_list(seq_msa_select), dtype=tf.float32) # (512, 174)
        seq_msa_select_extra_mask = tf.ones(shape_list(seq_msa_select_extra), dtype=tf.float32) # (34819, 174)



        """ (2) Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""
        # Make agreement score as weighted Hamming distance

        sample_one_hot = tf.one_hot(seq_msa_select, 23) # (512, 174, 23)
        extra_one_hot = tf.one_hot(seq_msa_select_extra, 23) # (34819, 174, 23)

        num_seq, num_res, _ = shape_list(sample_one_hot)
        extra_num_seq, _, _ = shape_list(extra_one_hot)

        # Determine how much weight we assign to each agreement.  In theory, we could
        # use a full blosum matrix here, but right now let's just down-weight gap
        # agreement because it could be spurious.
        # Never put weight on agreeing on BERT mask
        gap_agreement_weight=0.

        weights = tf.concat([
          tf.ones(21),
          gap_agreement_weight * tf.ones(1),
          np.zeros(1)], 0)

        # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
        # in an optimized fashion to avoid possible memory or computation blowup.
        agreement = tf.matmul(
          tf.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
          tf.reshape(sample_one_hot * weights, [num_seq, num_res * 23]),
          transpose_b=True) # shape=(34819, 512)

        extra_cluster_assignment = tf.argmax(agreement, axis=1, output_type=tf.int32) # shape=(34819,)
        extra_cluster_assignment


        """ (3) Produce profile and deletion_matrix_mean within each cluster."""
        def csum(x,extra_cluster_assignment,num_seq):
            return tf.math.unsorted_segment_sum(
                x, extra_cluster_assignment, num_seq)

        num_seq = shape_list(seq_msa_select)[0]
        mask_counts = 1e-6 + seq_msa_select_mask + csum(seq_msa_select_extra_mask,extra_cluster_assignment,num_seq)  # Include center

        msa_sum = csum(seq_msa_select_extra_mask[:, :, None] * tf.one_hot(seq_msa_select_extra, 23),extra_cluster_assignment,num_seq)
        msa_sum += tf.one_hot(seq_msa_select, 23)  # Original sequence
        msa_cluster_profile = msa_sum / mask_counts[:, :, None]

        del msa_sum

        """ (4) MSA features are cropped so only `max_extra_msa` sequences are kept."""
        num_seq_extra = tf.shape(seq_msa_select_extra)[0]
        num_sel = tf.minimum(max_extra_msa, num_seq_extra)
        select_indices = tf.compat.v1.random_shuffle(tf.range(0, num_seq_extra))[:num_sel]
        extra_msa_cropped = tf.gather(seq_msa_select_extra, select_indices)
        seq_msa_extra_mask_cropped = tf.gather(seq_msa_select_extra_mask, select_indices)

        """ (5) Create and concatenate MSA features."""
        msa_1hot = tf.one_hot(seq_msa_select, 23, axis=-1)
        msa_extra_1hot = tf.one_hot(extra_msa_cropped, 23, axis=-1)

        msa_feat = [msa_1hot]
        msa_extra_feat = [msa_extra_1hot]
        msa_feat.extend([msa_cluster_profile])


        msa_extra_feat_encode = tf.concat(msa_extra_feat, axis=-1)
        msa_feat_encode = tf.concat(msa_feat, axis=-1)
        
        return msa_feat_encode,msa_extra_feat_encode
    
    def check_sequences(self, feature):
        for i in range(0,len(feature['protein_id'])):
            pid = feature['protein_id'][i]
            aatype = feature['target_feat'][i]
            with open('example_data/sequences/' + pid + '.fasta') as train_f: 
                    seq = train_f.read().strip('\n').split('\n')[1]
            if seq != msa.onehot_to_sequence(aatype):
                print(pid,' has incorrect sequence encoding')
            else:
                print(pid,' pass')
    
    def visualize_msa_sequence(self, msas_encoding, pid):
        with open('example_data/msa/' + pid + '.aln') as train_f: 
            aln_sample = train_f.read().strip('\n').split('\n')
        
        msa_encodes = msas_encoding[:,:,0:23]  #(512, 174, 46)

        # Deduplicate but preserve order (hence can't use set).

        deduped_full_single_chain_msa = []   
        for i in range(0,len(msa_encodes)):
            seq_onehot = msa_encodes[i] # (174, 23)
            seq = msa.onehot_to_msa(seq_onehot)
            if seq not in aln_sample:
                print('alignment not found in ',pid, '\n', seq)
            deduped_full_single_chain_msa.append(seq)

        total_msa_size = len(deduped_full_single_chain_msa)
        print(f'\n{total_msa_size} unique sequences found in total for sequence '
            f'{pid}\n')

        aa_map = {res: i for i, res in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')}
        msa_arr = np.array(
          [[aa_map[aa] for aa in seq] for seq in deduped_full_single_chain_msa])

        plt.figure(figsize=(12, 3)) 
        plt.title(f'Per-Residue Count of Non-Gap Amino Acids in the MSA for Sequence '
                f'{pid}')
        plt.plot(np.sum(msa_arr != aa_map['-'], axis=0), color='black')
        plt.ylabel('Non-Gap Count')
        plt.yticks(range(0, total_msa_size + 1, max(1, int(total_msa_size / 3))))
        plt.show()    
        
    def get_feature_and_y_gcn(self, data_subset, contact_comb = 0):
        #target_model = 'FLOUDAS_SERVER_TS1'
        subset = data_subset.sort_values(by = 'ResId') 
        subst_pssm = subset['pssm'].str.split(',',expand=True).to_numpy().astype(np.float32)  
        subst_rosetta = subset['rosetta'].str.split(',',expand=True).to_numpy().astype(np.float32)  
        subst_ss = subset['ss'].str.split(',',expand=True) 
        subst_lddt = subset['Ca_lddt']
        subst_gdt = subset['GDT'].unique()
        subst_coord = subset[['x','y','z']]

        ## apply sigmoid to pssm
        subst_pssm = 1/(1 + np.exp(-subst_pssm))

        ## apply sigmoid to rosetta?
        subst_rosetta = 1/(1 + np.exp(-subst_rosetta))

        # Compute distance matrix
        model_distance = squareform(pdist(subst_coord, metric='euclidean'))

        ## reset index in case we have missing residues 
        subset['ResId_new'] = subset.reset_index().index.to_list()
        model_L = len(subset)
        ResIdMap = dict(zip(subset.ResId, subset.ResId_new))

        L5_short_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_short']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L5_short_contact[aa1_new,aa2_new] = 1
              L5_short_contact[aa2_new,aa1_new] = 1

        L5_medium_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_medium']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L5_medium_contact[aa1_new,aa2_new] = 1
              L5_medium_contact[aa2_new,aa1_new] = 1

        L5_long_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_long']
            if aa2 != -1:
              #print(aa1,aa2)
              try: 
                aa1_new = ResIdMap[aa1]
                aa2_new = ResIdMap[aa2]
              except:
                print('Failed to find aa')
                print('aa1:',aa1)
                print('aa2:',aa2)
                print('id_model:',id_model)
              L5_long_contact[aa1_new,aa2_new] = 1
              L5_long_contact[aa2_new,aa1_new] = 1

        L2_short_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_short']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L2_short_contact[aa1_new,aa2_new] = 1
              L2_short_contact[aa2_new,aa1_new] = 1

        L2_medium_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_medium']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L2_medium_contact[aa1_new,aa2_new] = 1
              L2_medium_contact[aa2_new,aa1_new] = 1


        L2_long_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_long']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L2_long_contact[aa1_new,aa2_new] = 1
              L2_long_contact[aa2_new,aa1_new] = 1

        L_short_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_short']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L_short_contact[aa1_new,aa2_new] = 1
              L_short_contact[aa2_new,aa1_new] = 1

        L_medium_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_medium']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L_medium_contact[aa1_new,aa2_new] = 1
              L_medium_contact[aa2_new,aa1_new] = 1


        L_long_contact = np.zeros((model_L, model_L))
        for ind in subset.index:
            aa1 = subset.loc[ind,'ResId']
            aa2 = subset.loc[ind,'L5_long']
            if aa2 != -1:
              #print(aa1,aa2)
              aa1_new = ResIdMap[aa1]
              aa2_new = ResIdMap[aa2]
              L_long_contact[aa1_new,aa2_new] = 1
              L_long_contact[aa2_new,aa1_new] = 1


        if contact_comb == 1:
            all_contact = L5_short_contact + L5_medium_contact + L5_long_contact + L2_short_contact + L2_medium_contact + L2_long_contact + L_short_contact + L_medium_contact + L_long_contact
        elif contact_comb == 2:
            all_contact = L_short_contact + L_medium_contact + L_long_contact
        elif contact_comb == 3:
            all_contact = L2_short_contact + L2_medium_contact + L2_long_contact
        elif contact_comb == 4:
            all_contact = L5_short_contact + L5_medium_contact + L5_long_contact
        elif contact_comb == 5:
            all_contact = L5_long_contact
        elif contact_comb == 6:
            all_contact = L_long_contact
        else:
            all_contact = L5_short_contact + L5_medium_contact + L5_long_contact + L2_short_contact + L2_medium_contact + L2_long_contact + L_short_contact + L_medium_contact + L_long_contact


        all_contact[all_contact>0] = 1
        all_contact = all_contact.astype(np.uint8)

        # set diagnol to 1?
        np.fill_diagonal(all_contact, 1)

        model_distance = np.multiply(all_contact,model_distance)

        return subst_rosetta,subst_pssm,subst_ss.to_numpy().astype(np.float32),model_distance.astype(np.float32),all_contact.astype(np.float32), subst_lddt.to_numpy().astype(np.float32),subst_gdt[0].astype(np.float32)



if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='EXAMPLE:\npython3 /faculty/jhou4/Projects/protein_folding/train_network_Nov28_SingleTrain.py  -b 5 -r 0-50 -n -1 -c 64 -e 10 -d 4 -f 8 -p /faculty/jhou4/Projects/protein_folding/data/Own_data/Sharear -v 0 -o /faculty/jhou4/Projects/protein_folding/train_results_20201128_len0_50_4')

    parser.add_argument('-w', type=str, required = True, dest = 'work_dir', help="working directory")
    parser.add_argument('-f', type=str, required = True, dest = 'feature_file', help="feature_file")
    parser.add_argument('-m', type=str, required = True, dest = 'model_dir', help="model_dir")
    
    args = parser.parse_args()
    return args


args = get_args()

work_dir                   = args.work_dir #
feature_file                   = args.feature_file # 
model_dir               = args.model_dir



# load model
contact_comb = 1
loaded = True

model_translator = tf.saved_model.load(model_dir)


eva_generator = ProteinQAData_Generator(work_dir, feature_file,batch_size='single')


pred2lddt_prediction = {}
# load models per target 
data = pd.read_csv(feature_file,sep='\t',header=0)

target = data['target'].unique()[0]
comb = target +  '|' + data["model"].unique()


target_batch_size = 1
batch_len = int(len(comb) /target_batch_size)+1
for indx in range(0,batch_len):
    batch_start = indx * target_batch_size
    batch_end = (indx+1) * target_batch_size
    if batch_end <= len(comb):
        batch_list = comb[batch_start:batch_end]
    else:
        batch_list = comb[batch_start:]
    
    if len(batch_list)==0:
        continue
    
    print("batch_list: ",batch_list)
    start_time = time.time()
    print("start processing: ",batch_list)
    target_feat_batch, msa_feat_batch, pdb_rosetta_batch, seq_pssm_batch, seq_ss_batch, seq_contact_batch, pdb_distance_pair_batch, model_aa_lddt_batch, model_gdt_batch  = eva_generator.collect_data(batch_list)
    print("--- feature processing %s seconds ---" % (time.time() - start_time))

    
    start_time = time.time()
    if loaded: 
        pred_logits = model_translator.tf_translate(target_feat_batch, msa_feat_batch, pdb_rosetta_batch, seq_pssm_batch, seq_ss_batch, seq_contact_batch, pdb_distance_pair_batch)
    else: 
        pred_logits = model_translator(target_feat_batch, msa_feat_batch, pdb_rosetta_batch, seq_pssm_batch, seq_ss_batch, seq_contact_batch, pdb_distance_pair_batch, target_label =model_aa_lddt_batch)
    print("--- prediction %s seconds ---" % (time.time() - start_time))
    
    
    pred_logits = tf.clip_by_value(pred_logits['outputs'], clip_value_min=0, clip_value_max=1)
    
    
    for model_idx in range(0,len(batch_list)):
        #id_model = 'T0860|FLOUDAS_SERVER_TS1|136'
        model_name = batch_list[model_idx]
        node_prob = tf.reshape(pred_logits[model_idx],[-1])
        
        model_pred_lddt_global = np.mean(node_prob)   
        
        
        # model_name: model_name:  T0949|slbio_server_TS5
        #print("model_name: ",model_name)
        targetname = model_name.split('|')[0]
        modelid = model_name.split('|')[1]
        
        
        if targetname in pred2lddt_prediction:
            pred2lddt_prediction[targetname] = pred2lddt_prediction[targetname] + "\n" + modelid + " " + str(np.round(model_pred_lddt_global,5))
            for lddt in node_prob:
                pred2lddt_prediction[targetname] = pred2lddt_prediction[targetname] + " " + str(np.round(lddt,5))
        else:
            pred2lddt_prediction[targetname] = modelid + " " + str(np.round(model_pred_lddt_global,5))
            for lddt in node_prob:
                pred2lddt_prediction[targetname] = pred2lddt_prediction[targetname] + " " + str(np.round(lddt,5))
                
        
for target in pred2lddt_prediction:
    myfile = open(work_dir+'/' + target + '.pred_qalddt', 'w')
    myfile.write(pred2lddt_prediction[target])
    print(target,"->",pred2lddt_prediction[target])
    myfile.close()
    print("Saved to ", work_dir+'/' + target + '.pred_qalddt')









