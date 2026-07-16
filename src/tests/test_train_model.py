from pathlib import Path

from src.train_model import train_model


def test_train_model_uses_accuracy_defaults_and_returns_best_model(
        monkeypatch, tmp_path):
    captured = {}

    class FakeResults:
        save_dir = tmp_path / "runs" / "detect" / "train"
        results_dict = {
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.71,
        }

    class FakeYOLO:
        def __init__(self, weights):
            captured["weights"] = weights

        def train(self, **kwargs):
            captured.update(kwargs)
            weights_dir = FakeResults.save_dir / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_bytes(b"model")
            return FakeResults()

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YOLO_MODEL", raising=False)
    monkeypatch.setattr("src.train_model.YOLO", FakeYOLO)
    monkeypatch.setattr("src.train_model.get_optimal_batch_size", lambda: 4)
    monkeypatch.setattr("src.train_model.torch.cuda.is_available", lambda: False)

    model_path = train_model("data.yaml")

    assert captured["weights"] == "yolov8m.pt"
    assert captured["epochs"] == 75
    assert captured["device"] == "cpu"
    assert Path(model_path).resolve() == (
        FakeResults.save_dir / "weights" / "best.pt").resolve()
