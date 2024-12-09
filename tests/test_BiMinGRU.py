import torch
from layers.rnn.BiMinGRU import BiMinGRU


class TestBiMinGRU:
    def test_batched_forward(self):
        batch_size, hidden_size = 2, 10
        seq1_len, seq2_len = 10, 6
        layer = BiMinGRU(hidden_size, hidden_size)
        layer.eval()
        x = torch.randn((batch_size, seq1_len, hidden_size))

        # compute batched
        mask = torch.zeros((batch_size, seq1_len))
        mask[1,seq2_len:] = 1
        mask = mask.bool()
        o = layer(x, mask)
        assert o.shape == (batch_size, seq1_len, hidden_size)
        seq1_hidden_batched, seq2_hidden_batched = o[0][:seq1_len], o[1][:seq2_len]

        # compute sequential
        seq1 = x[0][:seq1_len].unsqueeze(0)
        seq2 = x[1][:seq2_len].unsqueeze(0)
        seq1_hidden_sequential = layer(seq1)[0]
        seq2_hidden_sequential = layer(seq2)[0]

        assert seq1_hidden_sequential.shape == seq1_hidden_batched.shape == (seq1_len, hidden_size)
        assert seq2_hidden_sequential.shape == seq2_hidden_batched.shape == (seq2_len, hidden_size)
        assert torch.allclose(seq1_hidden_batched, seq1_hidden_sequential, rtol=1e-4, atol=1e-6)
        assert torch.allclose(seq2_hidden_batched, seq2_hidden_sequential, rtol=1e-4, atol=1e-6)



