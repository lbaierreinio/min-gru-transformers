from src.minGRU.ParallelMinGRU import ParallelMinGRU
import torch
class TestMinGRU:
    def test_min_gru(self):
        batch_size, seq_len, input_size, hidden_size = 1, 2, 2, 3
        parallel_min_gru = ParallelMinGRU(input_size, hidden_size)
        x = torch.randn((batch_size, seq_len, input_size))
        h_0 = torch.randn((batch_size, 1, hidden_size))

        h_parallel = parallel_min_gru(x, h_0)

        assert h_parallel.shape == (batch_size, seq_len, input_size)
