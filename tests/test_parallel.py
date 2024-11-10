from src.utils.utility import parallel_scan_log
import torch
class TestParallel:
    def test_parallel_scan_log(self):
        batch_size, seq_len, hidden_size = 1, 2, 3 # TODO: Test case where batch_size > 1
        a = torch.square(torch.randn((batch_size, seq_len, hidden_size)))
        b = torch.square(torch.randn((batch_size, seq_len+1, hidden_size)))

        # Parallel Computation
        h_parallel = parallel_scan_log(a, b)

        h_sequential = [b[0][0]]
        for a_i, b_i in zip(a[0], b[0][1:]):
            h_sequential.append(a_i*h_sequential[-1] + b_i)
        
        # Reshape to match the output of parallel_scan_log
        h_sequential = torch.stack(h_sequential[1:]).unsqueeze(0)

        # Assertion
        assert torch.allclose(h_parallel, h_sequential, atol=1e-5)
