from layers.rnn.MinGRU import MinGRU
import torch


class TestMinGRU:
    def test_min_gru_modes(self):
        # Test parallel mode
        layer = MinGRU(3, 5)
        batch = torch.randn(4, 10, 3) # BATCH_SIZE, SEQ_LEN, DIM_IN
        mask = torch.zeros((4, 10)).bool()
        mask[0, 4:] = 1
        mask[1, 3:] = 1
        output_parallel = layer(batch, mask=mask, is_sequential=False)
        
        # Test sequential mode
        output_sequential = layer(batch, mask=mask, is_sequential=True)
        assert torch.allclose(output_parallel, output_sequential)
