import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentCLAPText(nn.Module):
    """Minimal student text encoder (mirrors tinyCLAP, can be replaced with a small transformer)."""
    def __init__(self, embedding_dim=512, vocab_size=30522, hidden_dim=256, num_layers=2, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.transpose(0, 1)  # (seq, batch, hidden)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, hidden)
        x = self.projection(x)
        x = F.normalize(x, dim=-1)
        return x

    def export_to_onnx(self, output_path: str, device='cpu'):
        """Export the student text model to ONNX format."""
        self.eval()
        seq_len = 77  # CLAP uses max_length=77
        batch_size = 1
        dummy_input_ids = torch.randint(0, self.embedding.num_embeddings, (batch_size, seq_len), dtype=torch.long).to(device)
        dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
        torch.onnx.export(
            self,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['embedding'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                'embedding': {0: 'batch_size'}
            },
            opset_version=15,
        )
        print(f"✅ Exported StudentCLAPText to ONNX: {output_path}")
