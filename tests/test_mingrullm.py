from src.lm.minGRULM import minGRULM
import torch

class TestMinGRULM:
    def test_mingru_lm(self):
        batch_size = 2
        seq_len = 3
        hidden_dim = 4
        input_dim=2
        num_tokens = 10
        x = torch.randint(0, num_tokens, (batch_size, seq_len), dtype=torch.long)
        h_0 = torch.zeros((batch_size, 1, hidden_dim))

        mingru_lm = minGRULM(
            num_tokens=num_tokens,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

        out = mingru_lm(x, h_0)

        assert out.shape == (batch_size, seq_len, hidden_dim)
