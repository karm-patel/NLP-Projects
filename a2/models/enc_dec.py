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

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.MAX_LENGTH = 7
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        target , encoder_op, _ = inputs
        # target.shape = [B, L]
        # encoder_op.shape = [1, B, H)]
        
        i = 0
        decoder_logits= []
        decoder_ip_h = encoder_op
        decoder_ip_x = target[:,0]
        for i in range(self.MAX_LENGTH):
            
            # forward step
            decoder_ip_x = F.relu(self.embedding(decoder_ip_x)) # [B, D]
            decoder_ip_x = torch.unsqueeze(decoder_ip_x, dim=1) # [B, 1, D]
            all_ops, op = self.gru(decoder_ip_x, decoder_ip_h) # [B, 1, H], [1, B, H]
            op = torch.squeeze(op) # [B,H]
#             op = F.relu(op)
            logits = self.linear(op) # [B, output_size]
               
            decoder_logits.append(logits)
            
            
            decoder_ip_h = torch.permute(all_ops, (1,0,2)) # [1, B, H]  
            _, decoder_ip_x = torch.max(logits, dim=-1) # [B,1]
#             print(decoder_ip_x.shape, decoder_ip_h.shape)
        
        decoder_logits = torch.stack(decoder_logits, dim=1) # [B, 7, output_size]
        log_probs = F.log_softmax(decoder_logits, dim=-1) 
        return log_probs, decoder_logits, None