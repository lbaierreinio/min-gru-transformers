import torch


class TestParallel:
    def test_parallel(self):
        return  # TODO: Re-implement test case
        # TODO: Test case where batch_size > 1
        batch_size, seq_len, hidden_size = 1, 2, 3
        a = torch.abs(torch.randn((batch_size, seq_len, hidden_size)))
        b = torch.abs(torch.randn((batch_size, seq_len, hidden_size)))

        log_a = torch.log(a)
        log_b = torch.log(b)

        # Parallel Computation
        h_parallel = parallel_scan_log(log_a, log_b)

        h_sequential = [b[0][0]]
        for a_i, b_i in zip(a[0][1:], b[0][1:]):
            h_sequential.append(a_i*h_sequential[-1] + b_i)

        # Reshape to match the output of parallel_scan
        h_sequential = torch.stack(h_sequential).unsqueeze(0)

        # Assertion
        assert torch.allclose(h_parallel, h_sequential, atol=1e-5)
