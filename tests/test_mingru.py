from layers.rnn.MinGRU import MinGRU
import torch

class TestMinGRU:
    def test_min_gru_modes(self):
        batch_size, seq_len, embedding_dim = 2, 8, 4

        model = MinGRU(dim_in=embedding_dim, dim_hidden=embedding_dim)
        model.eval()

        x = torch.randn((batch_size, seq_len, embedding_dim))

        h_prev = torch.zeros((batch_size, embedding_dim))
        out_sequential = torch.zeros((batch_size, seq_len, embedding_dim))
        for i in range(seq_len):
            h_prev = model(x[:, i], h_prev=h_prev)
            out_sequential[:, i] = h_prev
        
        out_parallel = model(x)

        assert torch.allclose(out_sequential, out_parallel, rtol=1e-4, atol=1e-6)
