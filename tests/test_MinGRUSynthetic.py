from models.MinGRUSynthetic import MinGRUSynthetic
import torch


class TestMinGRUSynthetic:
    def test_min_gru_synthetic(self):
        batch_size, seq_len, embedding_dim = 2, 8, 4

        model = MinGRUSynthetic(vocab_size=100, embedding_dim=embedding_dim, num_classes=4, num_layers=2, bidirectional=False)
        model.eval()

        x = torch.randint(0, 100, (batch_size, seq_len))

        out_seq = model(x, is_sequential=True)
        out_parallel = model(x, is_sequential=False)

        assert torch.allclose(out_seq, out_parallel, rtol=1e-4, atol=1e-6)
    
    def test_min_gru_synthetic_mask(self):
        batch_size, seq_len, embedding_dim = 2, 8, 4

        model = MinGRUSynthetic(vocab_size=100, embedding_dim=embedding_dim, num_classes=4, num_layers=2, bidirectional=False)
        model.eval()

        x = torch.randint(0, 100, (batch_size, seq_len))
        mask = torch.zeros((batch_size, seq_len))
        mask[0, 2:] = 1
        mask[1, 4:] = 1
        mask = mask.bool()

        out_seq = model(x, is_sequential=True, mask=mask)
        out_parallel = model(x, is_sequential=False, mask=mask)

        assert torch.allclose(out_seq, out_parallel, rtol=1e-4, atol=1e-6)



