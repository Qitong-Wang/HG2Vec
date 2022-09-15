import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
import random
import os

class HG2VecDataset(Dataset):
    def __init__(self, args):
        # Input file
        self.input_vector_folder = args.input_vector_folder
        self.files_names = os.listdir(self.input_vector_folder)
        self.files_names = sorted(self.files_names)
        self.lines = []
        self.routes = 1
        self.path_length = 1
        self.lr=  args.lr
        self.file_idx = 0
     
    
        # For sampler
        self.id_info = pd.read_csv(args.input_id_info, header=None)
        self.window_size = args.window
        self.NEGATIVE_TABLE_SIZE = 1e8
        self.negatives = []
        self.init_neg_table()
        self.negpos = 0

        pair_file = open(args.strong_file, "rb")
        self.strong_dict = pickle.load(pair_file)
        pair_file.close()
        pair_file = open(args.weak_file, "rb")
        self.weak_dict = pickle.load(pair_file)
        pair_file.close()
        pair_file = open(args.syn_file, "rb")
        self.syn_dict = pickle.load(pair_file)
        pair_file.close()
        pair_file = open(args.ant_file, "rb")
        self.ant_dict = pickle.load(pair_file)
        pair_file.close()

        self.neg_size = args.neg_size
        self.strong_size = args.strong_size
        self.weak_size = args.weak_size
        self.syn_size = args.syn_size
        self.ant_size = args.ant_size

        #self.v_start = 0
        self.neg_start = 0
        self.strong_start = self.neg_start + self.window_size * self.neg_size
        self.weak_start = self.strong_start + self.window_size * self.strong_size
        self.syn_start = self.weak_start + self.window_size * self.weak_size
        self.ant_start = self.syn_start + self.window_size * self.syn_size
        self.idx_end = self.ant_start + self.window_size * self.ant_size
        
       
        #Masks
        self.sig_mask = torch.ones(self.idx_end).int()
        self.sig_mask[:self.strong_start] = -1
        self.sig_mask[self.ant_start:] = -1

        self.score_mask = torch.ones(self.idx_end).float()
        self.score_mask[:self.strong_start] = args.beta_neg
        self.score_mask[self.strong_start:self.weak_start] = args.beta_strong
        self.score_mask[self.weak_start:self.syn_start] = args.beta_weak
        self.score_mask[self.syn_start:self.ant_start] = args.beta_syn
        self.score_mask[self.ant_start:] = args.beta_ant
      
        self.dropout = args.dropout
        self.update_dataset(args)


    def get_single_item(self, line, idx):
        u = line[idx]
        context_array = np.zeros(self.window_size)
     
        result_array = np.zeros(self.idx_end).astype(int)
        min_c = max(0, idx - int(self.window_size // 2))
        max_c = min(self.path_length-1, idx + int(self.window_size // 2))
        # index of context in the line
        contexts = np.append(line[min_c:idx], line[idx+1:max_c+1]) #Skip when context == target

        c_length = len(contexts) + 1
        # assign u
        u_array = np.array(u)
        # assign c
        context_array[0] = line[idx]
        context_array[1:c_length] = contexts


        for c_step in range(0,c_length): # c_length_idx: 0,1,2,3
            # assign c
            c = context_array[c_step]
            # assign negative
            start = c_step*self.neg_size
            result_array[start:start+self.neg_size]= self.get_negatives(c, self.neg_size)
            start = self.strong_start + c_step* self.strong_size
            result_array[start:start+ self.strong_size] = self.get_pairs(c,"strong", self.strong_size)
            start = self.weak_start + c_step * self.weak_size
            result_array[start:start+self.weak_size] = self.get_pairs(c, "weak", self.weak_size)
            start = self.syn_start + c_step * self.syn_size
            result_array[start:start+self.syn_size] = self.get_pairs(c, "syn", self.syn_size)
            start = self.ant_start + c_step * self.ant_size
            result_array[start:start+self.ant_size] = self.get_pairs(c, "ant", self.ant_size)
        return u_array, context_array, result_array
     
    def get_line_item(self, idx):
        target_array = self.lines[idx]
        half_window = self.window_size//2
        #https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray
        context_indexer =  np.arange(self.window_size)[None, :] + np.arange(self.path_length)[:, None] - half_window
        context_indexer[context_indexer>=self.path_length] = 0 
        context_array = target_array[context_indexer]

        for i in range(0,half_window):
            for j in range(0,half_window-i):
                context_array[i,j] = 0
        for i in range(0,half_window):
            for j in range(0,half_window-i):
                context_array[self.path_length-1-i,self.window_size-1-j] = 0
        
       
        info_array = np.zeros((self.path_length,self.idx_end)).astype(int)

        for i in range(self.path_length):
            for c_step in range(self.window_size): # c_length_idx: 0,1,2,3
                # assign c
                c = context_array[i,c_step]
                if c == 0:
                    continue
                # assign negative
                start = c_step*self.neg_size
                info_array[i,start:start+self.neg_size]= self.get_negatives(c, self.neg_size)
                start = self.strong_start + c_step* self.strong_size
                info_array[i,start:start+ self.strong_size] = self.get_pairs(c,"strong", self.strong_size)
                start = self.weak_start + c_step * self.weak_size
                info_array[i,start:start+self.weak_size] = self.get_pairs(c, "weak", self.weak_size)
                start = self.syn_start + c_step * self.syn_size
                info_array[i,start:start+self.syn_size] = self.get_pairs(c, "syn", self.syn_size)
                start = self.ant_start + c_step * self.ant_size
                info_array[i,start:start+self.ant_size] = self.get_pairs(c, "ant", self.ant_size)

        target_array = target_array.reshape(-1,1)
        
        return target_array,context_array,info_array

    # initialize negative table
    # https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/data_reader.py
    def init_neg_table(self):
        pow_frequency = np.array(self.id_info.iloc[:,1]) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * self.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negatives(self, v, size):
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            response = np.concatenate((response, self.negatives[0:self.negpos]))

        return response

    def get_pairs(self, v, pairs, size):
        response = np.zeros(size)
        if pairs == "strong":
            target_dict = self.strong_dict
        elif pairs == "weak":
            target_dict = self.weak_dict
        elif pairs == "syn":
            target_dict = self.syn_dict
        else:
            target_dict = self.ant_dict
        if v in target_dict.keys(): #Contain pairs
            pairs_value = target_dict[v]
            pair_length = len(pairs_value)
            if size < pair_length: # Enough pairs
                response = np.random.choice(pairs_value,size, replace=False)
            else: #Not enough pairs
                response[:pair_length] = pairs_value

        return response
    
    def __len__(self):     
        return self.routes

    def __getitem__(self, idx):
        return self.get_line_item(idx)

    # update the dataset for a new epoch
    def update_dataset(self,args):
        idx = int(self.file_idx % (len(self.files_names)/2))
    
        file_name = self.files_names[2*idx]
        file_name_2 = self.files_names[2*idx+1]
      
        lines = np.load(self.input_vector_folder + file_name)
        lines_2 = np.load(self.input_vector_folder + file_name_2)
        self.lines = np.concatenate((lines,lines_2),axis=0)
        self.routes = self.lines.shape[0]
        self.path_length = self.lines.shape[1]

        self.file_idx += 1

    