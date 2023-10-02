import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # input.shape = (N, Batch, Hidden)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        
    def forward(self, enc_all_ops, dec_h_so_far):
        # enc_all_ops = (B, N, H)
        # dec_h_so_far = (B, 1, H)
        scores = self.Va(torch.tanh(self.Ua(enc_all_ops) + self.Wa(dec_h_so_far))) # (B, N, 1)
        scores = scores.squeeze(2).unsqueeze(1) # [B, 1, N]
        weights = F.softmax(scores, dim=-1) # (B, 1, N)
        context = torch.bmm(weights, enc_all_ops) # (B, 1, H)
        return context, weights 
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.MAX_LENGTH = 7
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_size, output_size)
        self.attention = BahdanauAttention(hidden_size=hidden_size)
        
    def forward(self, inputs):
        target , encoder_op, enc_all_ops = inputs # [B, 7], [1, B, H], [B, N, H]
        
        
        decoder_logits= []
        decoder_ip_h = encoder_op
        decoder_ip_x = target[:,0]
        for i in range(self.MAX_LENGTH):
            
            # forward step
            decoder_ip_x = F.relu(self.embedding(decoder_ip_x)) # [B, H]
            decoder_ip_x = torch.unsqueeze(decoder_ip_x, dim=1) # [B, 1, H]
            context, weights = self.attention(enc_all_ops, decoder_ip_x) # [B, 1, H], [B, 1, 7]
            all_ops, op = self.gru(torch.cat((decoder_ip_x, context), dim=-1), decoder_ip_h) # [B, 1, 2*H], [1, B, H]
            op = torch.squeeze(op) # [B,H]
            logits = self.linear(op) # [B, output_size]
            decoder_logits.append(logits)
            
            
            decoder_ip_h = torch.permute(all_ops, (1,0,2)) # [1, B, H]  
            _, decoder_ip_x = torch.max(logits, dim=-1) # [B,1] -- argmax
        
        decoder_logits = torch.stack(decoder_logits, dim=1) # [B, 7, output_size]
        log_probs = F.log_softmax(decoder_logits, dim=-1) 
        return log_probs, decoder_logits, weights