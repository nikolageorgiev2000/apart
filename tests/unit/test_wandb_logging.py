from pathlib import Path

from apart.artifacts.logging import RunLogger


def test_wandb_offline_logs_config_metrics_and_summary(tmp_path: Path) -> None:
    logger = RunLogger(
        tmp_path,
        tensorboard=False,
        wandb_options={
            "enabled": True,
            "project": "apart-tests",
            "entity": None,
            "group": "test",
            "name": "offline-test",
            "mode": "offline",
            "tags": ["unit"],
        },
        run_config={"method": {"name": "test"}},
    )
    assert logger.wandb_run_id
    logger.log({"train/loss": 1.25}, step=1)
    logger.log_summary({"eval/activation_gap": 0.5})
    logger.close()
    assert list(tmp_path.glob("wandb/offline-run-*"))
