from models.minGRULM import MinGRULM
import torch


class TestMinGRULM:
    def test_min_gru_sequential(self):
        num_tokens = 10
        input_dim = 2
        hidden_dim = 4
        num_layers = 2
        x_t = torch.randint(0, num_tokens, (1,), dtype=torch.long)
        h_prev = torch.randn((num_layers, 1, hidden_dim))

        mingru_lm = MinGRULM(
            num_tokens=num_tokens,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        out, h_next = mingru_lm(x_t, h_prev)

        assert out.shape == (1, hidden_dim)  # Embedding of x_t
        assert len(h_next) == num_layers  # Hidden state of x_t
        for h in h_next:
            assert h.shape == (1, hidden_dim)

    def test_min_gru_parallel(self):
        batch_size = 2
        seq_len = 3
        hidden_dim = 4
        input_dim = 2
        num_tokens = 10
        num_layers = 2

        x = torch.randint(0, num_tokens, (batch_size,
                          seq_len), dtype=torch.long)

        mingru_lm = MinGRULM(
            num_tokens=num_tokens,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        out, _ = mingru_lm(x)

        assert out.shape == (batch_size, seq_len, hidden_dim)
