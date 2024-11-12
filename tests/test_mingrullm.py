from src.lm.minGRULM import MinGRULM
import torch

class TestMinGRULM:
    def test_mingru_lm(self):
        batch_size = 2
        seq_len = 3
        hidden_dim = 4
        input_dim=2
        num_tokens = 10
        num_layers = 2
        x = torch.randint(0, num_tokens, (batch_size, seq_len), dtype=torch.long)
        h_prev = torch.zeros((batch_size, num_layers, hidden_dim))

        mingru_lm = MinGRULM(
            num_tokens=num_tokens,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        out, h_next = mingru_lm(x, h_prev)

        assert out.shape == (batch_size, seq_len, hidden_dim)
