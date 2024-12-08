import torch
from layers.rnn.BiMinGRU import BiMinGRU


class TestBiMinGRU:
    def test_batched_forward(self):
        batch_size, hidden_size = 2, 10
        seq1_len, seq2_len = 10, 6
        layer = BiMinGRU(hidden_size, hidden_size)
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
        assert torch.allclose(seq1_hidden_batched, seq1_hidden_sequential)
        assert torch.allclose(seq2_hidden_batched, seq2_hidden_sequential)
    
    def test_bi_min_gru_modes(self):
        # Test parallel mode
        layer = BiMinGRU(2, 3)
        batch = torch.randn(2, 3, 2) # BATCH_SIZE, SEQ_LEN, DIM_IN
        #mask = torch.zeros((4, 10)).bool()
        #mask[0, 4:] = 1
        #mask[1, 3:] = 1
        output_parallel = layer(batch, is_sequential=False)
        
        # Test sequential mode
        output_sequential = layer(batch, is_sequential=True)
        assert torch.allclose(output_parallel, output_sequential)


