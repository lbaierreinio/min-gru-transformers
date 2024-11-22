from minGRU.minGRU import MinGRU
import torch
class TestMinGRU:
    def test_min_gru_parallel(self):
        batch_size, seq_len, input_size, hidden_size = 1, 2, 2, 3
        min_gru = MinGRU(input_size, hidden_size)
        x = torch.randn((batch_size, seq_len, input_size))

        o = min_gru(x)

        assert o.shape == (batch_size, seq_len, input_size)
    
    def test_min_gru_sequential(self):
        input_size, hidden_size = 2, 3
        min_gru = MinGRU(input_size, hidden_size)

        x_t = torch.randn((1, input_size))
        h_prev = torch.randn((1, hidden_size))

        o, h = min_gru(x_t, h_prev, return_hidden=True)

        assert o.shape == (1, input_size)
        assert h.shape == (1, hidden_size)