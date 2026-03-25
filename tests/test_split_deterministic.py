from gds.data.datasets import stratified_train_val_split


def test_stratified_split_deterministic() -> None:
    labels = [i % 10 for i in range(1000)]
    train_a, val_a = stratified_train_val_split(labels=labels, val_size=200, seed=42)
    train_b, val_b = stratified_train_val_split(labels=labels, val_size=200, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 800
    assert len(val_a) == 200

