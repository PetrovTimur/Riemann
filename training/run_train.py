import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    yaml_cfg = yaml.dump(resolved_cfg, sort_keys=False)
    print(yaml_cfg)

    trainer = hydra.utils.instantiate(cfg)(config=cfg)
    trainer.train()

if __name__ == "__main__":
    main()
