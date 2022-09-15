import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import einops



class HG2VecModel(nn.Module):

    def __init__(self, args, emb_size, emb_dimension, sig_mask, score_mask,context_mask):
        super(HG2VecModel, self).__init__()
        self.emb_size = emb_size
        self.sig_mask = sig_mask
        self.score_mask = score_mask
        self.context_mask = context_mask
       
        self.beta_pos = args.beta_pos
        self.emb_dimension = emb_dimension
        self.dropout = nn.Dropout(args.dropout)
        self.output_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True, padding_idx=0)
        self.input_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True, padding_idx=0)
        
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.output_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.input_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, info_v):
        emb_target_output = self.output_embeddings(pos_u) # b,l,1,d
        emb_context_input = self.input_embeddings(pos_v) # b,l,c,d
        emb_context_output = self.output_embeddings(pos_v) #b,l, c,d
        emb_info_input = self.input_embeddings(info_v) # b,l, i,d
        # dropout context
        emb_context_input = self.dropout(emb_context_input)
        emb_context_output = self.dropout(emb_context_output)
        emb_info_input = self.dropout(emb_info_input)
       
        emb_context_input = torch.einsum('blcd,c->blcd',emb_context_input,self.context_mask)
        emb_context_input = einops.rearrange(
            emb_context_input, 'b l c d -> b l d c'
        ) 
       
        emb_info_input = einops.rearrange(
            emb_info_input, 'b l i d -> b l d i'
        ) 
       
        score = torch.einsum('bltd,bldc->bltc', emb_target_output, emb_context_input)
       
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        score = torch.sum(score)
        
        info_score = torch.einsum('blcd,bldi->blci',emb_context_output, emb_info_input)
        info_score = torch.clamp(info_score, max=10, min=-10)
        info_score = torch.einsum('blci,i->blci',info_score,self.sig_mask)
        info_score = -F.logsigmoid(info_score)
        info_score = torch.einsum('blci,i->blci',info_score,self.score_mask)
        info_score = torch.sum(info_score)
  
        score = score + info_score
        return score



    def save_embedding(self, id2info, file_name):
        embedding = self.output_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2info), self.emb_dimension))
            for i in range(0,len(id2info)):
                e = ' '.join(map(lambda x: str(x), embedding[i]))
                f.write('%s %s\n' % (id2info.iloc[i,2], e))