from src.minGRU.minGRU import MinGRU
import torch
class TestMinGRU:
    def test_min_gru_parallel(self):
        batch_size, seq_len, input_size, hidden_size = 1, 2, 2, 3
        min_gru = MinGRU(input_size, hidden_size)
        x = torch.randn((batch_size, seq_len, input_size))
        h_0 = torch.randn((batch_size, 1, hidden_size))

        o, h = min_gru(x, h_0)

        assert o.shape == (batch_size, seq_len, input_size)
        assert h.shape == (batch_size, seq_len, hidden_size)
    
    def test_min_gru_sequential(self):
        batch_size, seq_len, input_size, hidden_size = 1, 1, 2, 3
        min_gru = MinGRU(input_size, hidden_size)
        x = torch.randn((batch_size, seq_len, input_size))
        h_0 = torch.randn((batch_size, 1, hidden_size))

        o, h = min_gru(x, h_0)

        assert o.shape == (batch_size, seq_len, input_size)
        assert h.shape == (batch_size, seq_len, hidden_size)