import argparse 
from dataset import HG2VecDataset
from model import HG2VecModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os 
import random
import numpy as np
 

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # Path
    argparser.add_argument('--output_vector_path', type=str, default="./hg2vec.txt",
                           help="The path of output vector file for evaluate")
    argparser.add_argument('--output_ckpt_path', type=str, default="./hg2vec.ckpt",
                           help="The path of output ckpt for further use")
    argparser.add_argument('--input_vector_folder', type=str, default="./path/",
                           help="The path of input path folder")
    argparser.add_argument('--input_id_info', type=str, default="./data/id_info.csv",
                           help="The path of input word2id folder")
    argparser.add_argument('--strong_file', type=str, default="./data/strong-pairs.pkl",
                           help="The path of input strong pairs")
    argparser.add_argument('--weak_file', type=str, default="./data/weak-pairs.pkl",
                           help="The path of input weak pairs")
    argparser.add_argument('--syn_file', type=str, default="./data/syn-pairs.pkl",
                           help="The path of input synonym pairs")
    argparser.add_argument('--ant_file', type=str, default="./data/ant-pairs.pkl",
                           help="The path of input antonym pairs")
    argparser.add_argument('--output_per_epoch', type=bool, default=False,
                           help="Whether store outputs each epoch")                    
    # Size
    argparser.add_argument('--window', type=int, default=5,
                           help="The size of window context")
    argparser.add_argument('--neg_size', type=int, default=5,
                           help="The size of negative sampling")
    argparser.add_argument('--strong_size', type=int, default=5,
                           help="The size of strong pairs sampling")
    argparser.add_argument('--weak_size', type=int, default=5,
                           help="The size of weak pairs sampling")
    argparser.add_argument('--syn_size', type=int, default=5,
                           help="The size of synonym pairs sampling")
    argparser.add_argument('--ant_size', type=int, default=5,
                           help="The size of antonym pairs sampling")

    argparser.add_argument('--batch_size', type=int, default=16,
                           help="The size of batch")
    argparser.add_argument('--emb_dimension', type=int, default=300,
                           help="The dimension of word embeddings")
    argparser.add_argument('--epochs', type=int, default=1,
                           help="The number of epochs")
    argparser.add_argument('--lr', type=float, default=0.003,
                           help="The learning rate")
    argparser.add_argument('--num_workers', type=int, default=8,
                           help="The number of workers")
    # Hyperparameter
    argparser.add_argument('--beta_pos', type=float, default=1.0,
                           help="The hyperparameter for the positive sampling")
    argparser.add_argument('--beta_neg', type=float, default=3.5,
                           help="The hyperparameter for the negative sampling")
    argparser.add_argument('--beta_strong', type=float, default=0.6,
                           help="The hyperparameter for the strong pairs")
    argparser.add_argument('--beta_weak', type=float, default=0.4,
                           help="The hyperparameter for the weak pairs")
    argparser.add_argument('--beta_syn', type=float, default=1.0,
                           help="The hyperparameter for the synonym pairs")
    argparser.add_argument('--beta_ant', type=float, default=1.0,
                           help="The hyperparameter for the antonym pairs")
    argparser.add_argument('--dropout', type=float, default=0.15,
                           help="The hyperparameter for the dropout rate")
    args = argparser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = HG2VecDataset(args)
    vocab_size = dataset.id_info.shape[0]
    
    sig_mask = dataset.sig_mask.to(device)
    score_mask = dataset.score_mask.to(device)
    context_mask = torch.zeros(args.window)
    context_mask[0] = 1
    context_mask = context_mask.to(device)
    model = HG2VecModel(args, vocab_size, args.emb_dimension, sig_mask, score_mask, context_mask).to(device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
               shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    model.train()
    start_idx = 0
    for epoch in range(start_idx,args.epochs):
        print("epoch")
        print(epoch)
  
        for i, (pos_u, pos_v,info_v) in enumerate(tqdm(train_loader)):             
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            info_v = info_v.to(device)

            optimizer.zero_grad()
            loss = model(pos_u, pos_v, info_v)

            loss.backward()
            optimizer.step()
      
        if (args.output_per_epoch): 
            output_name = args.output_vector_path[:-4]+"_epoch{:03d}".format(epoch)+args.output_vector_path[-4:]
            model.save_embedding(dataset.id_info,output_name)
            output_name = args.output_ckpt_path[:-5]+"_epoch{:02d}".format(epoch)+args.output_ckpt_path[-5:]
            torch.save(model.state_dict(), output_name)
        dataset.update_dataset(args)
    # Store after training
    if (not args.output_per_epoch): 
        torch.save(model.state_dict(),args.output_ckpt_path)
    model.save_embedding(dataset.id_info, args.output_vector_path)

