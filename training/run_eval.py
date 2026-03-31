import hydra
from omegaconf import DictConfig, OmegaConf
import yaml


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # Print resolved config for reproducibility
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    yaml_cfg = yaml.dump(resolved_cfg, sort_keys=False)
    print(yaml_cfg)

    trainer = hydra.utils.instantiate(cfg)(config=cfg)

    # Default behavior: run full validation (scalar loss + any module metrics).
    trainer.eval()

    # Optional: write per-row losses to a CSV table.
    # Usage: ++eval_output_table=eval_with_loss.csv (relative to hydra output dir)
    eval_out = cfg.get("eval_output_table", None)
    if eval_out:
        loss_col = cfg.get("eval_loss_col", "loss")
        add_preds = bool(cfg.get("eval_add_preds", True))
        pred_prefix = str(cfg.get("eval_pred_prefix", "pred_"))
        trainer.eval_to_table(
            output_path=str(eval_out),
            loss_col=str(loss_col),
            add_preds=add_preds,
            pred_prefix=pred_prefix,
        )


if __name__ == "__main__":
    main()

