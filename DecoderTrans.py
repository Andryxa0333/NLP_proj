import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens, n_embd, num_layers, num_heads, num_classes, max_seq_len=50):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, n_embd)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=n_embd, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(n_embd, num_classes)
        self.num_tokens = num_tokens
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len

    def forward(self, x, tgt=None):
        batch_size, seq_len = x.size()

        x_emb = self.embedding(x) + self.positional_encoding[:, :seq_len, :]

        if tgt is not None:
            tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(x.device)
        else:
            tgt_emb = x_emb
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        memory = x_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)

        decoded = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        decoded = decoded.transpose(0, 1)

        logits = self.fc_out(decoded)
        return logits

    @staticmethod
    def generate_square_subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask