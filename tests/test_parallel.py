from src.utils.utility import parallel_scan_log
import torch
class TestParallel:
    def test_parallel_scan_log(self):
        a = torch.tensor([0.1, 0.2, 0.3])
        b = torch.tensor([0.7, 0.8, 0.9, 1.1])
        x = parallel_scan_log(a, b)
        assert torch.allclose(x, torch.tensor([0.7000, 0.8700, 1.0740, 1.4222]), atol=1e-4)
