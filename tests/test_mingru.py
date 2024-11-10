from src.minGRU.ParallelMinGRU import ParallelMinGRU
import torch
class TestMinGRU:
    def test_min_gru(self):
        parallel_min_gru = ParallelMinGRU(2, 3)
        x = torch.randn((1, 2, 2))
        h_0 = torch.randn((1, 1, 3))

        h_parallel = parallel_min_gru(x, h_0)

        assert h_parallel.shape == (1, 2, 3)
