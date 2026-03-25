import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gds.scoring.forgetting import compute_forgetting_counts


class _TinyIndexedDataset:
    """Minimal dataset that yields (x, y, sample_id) like IndexedDataset."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, ids: torch.Tensor):
        self.x = x
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.ids[idx]


def test_forgetting_counts_non_negative() -> None:
    """All forgetting counts must be >= 0."""
    n = 40
    torch.manual_seed(0)
    x = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, 4, (n,))
    ids = torch.arange(n)

    ds = _TinyIndexedDataset(x, y, ids)
    loader = DataLoader(ds, batch_size=10, shuffle=True)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))

    fc, cum_loss, n_seen = compute_forgetting_counts(
        model=model,
        loader=loader,
        num_epochs=3,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        device=torch.device("cpu"),
        show_progress=False,
    )

    assert len(fc) == n
    for sid, count in fc.items():
        assert count >= 0, f"sample {sid} has negative forgetting count"

    # Verify cumulative loss and seen counts
    assert len(cum_loss) == n
    assert len(n_seen) == n
    for sid in fc:
        assert cum_loss[sid] >= 0.0, f"sample {sid} has negative cumulative loss"
        assert n_seen[sid] > 0, f"sample {sid} was never seen"


def test_easy_samples_have_fewer_forgetting_events() -> None:
    """Linearly separable samples should have fewer forgetting events
    than noisy/random-label samples."""
    torch.manual_seed(42)
    n = 60
    # Easy: class = sign of first feature (linearly separable)
    x_easy = torch.randn(n, 3, 4, 4)
    y_easy = (x_easy[:, 0, 0, 0] > 0).long()
    ids_easy = torch.arange(n)

    # Hard: same features, random labels
    x_hard = x_easy.clone()
    y_hard = torch.randint(0, 2, (n,))
    ids_hard = torch.arange(n)

    def _run(x, y, ids):
        ds = _TinyIndexedDataset(x, y, ids)
        loader = DataLoader(ds, batch_size=15, shuffle=True)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
        fc, cum_loss, n_seen = compute_forgetting_counts(
            model=model,
            loader=loader,
            num_epochs=10,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            device=torch.device("cpu"),
            show_progress=False,
        )
        return np.mean(list(fc.values()))

    easy_mean = _run(x_easy, y_easy, ids_easy)
    hard_mean = _run(x_hard, y_hard, ids_hard)
    assert easy_mean <= hard_mean, (
        f"Easy samples should have fewer forgetting events: "
        f"easy={easy_mean:.2f} vs hard={hard_mean:.2f}"
    )
