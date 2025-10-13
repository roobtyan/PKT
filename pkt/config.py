"""Configuration dataclasses and loading helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


@dataclass
class ComponentConfig:
    """Generic component configuration backed by a registry."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset split."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    shuffle: bool = False
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    additional_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    backbone: ComponentConfig
    head: ComponentConfig
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig(ComponentConfig):
    pass


@dataclass
class SchedulerConfig(ComponentConfig):
    pass


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    device: str = "cpu"
    log_interval: int = 10
    precision: str = "float32"
    output_dir: str = "outputs"
    gradient_accumulation: int = 1
    save_checkpoints: bool = True
    checkpoint_interval: int = 1
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    additional_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    datasets: Dict[str, DatasetConfig]
    dataloaders: Dict[str, DataLoaderConfig]
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    scheduler: SchedulerConfig | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "datasets": {k: vars(v) for k, v in self.datasets.items()},
            "dataloaders": {k: _dataloader_to_dict(v) for k, v in self.dataloaders.items()},
            "model": _component_to_dict(self.model),
            "optimizer": vars(self.optimizer),
            "scheduler": vars(self.scheduler) if self.scheduler else None,
            "trainer": _trainer_to_dict(self.trainer),
        }


def _component_to_dict(model_cfg: ModelConfig) -> Dict[str, Any]:
    return {
        "backbone": vars(model_cfg.backbone),
        "head": vars(model_cfg.head),
        "params": model_cfg.params,
    }


def _trainer_to_dict(cfg: TrainerConfig) -> Dict[str, Any]:
    return {
        "max_epochs": cfg.max_epochs,
        "device": cfg.device,
        "log_interval": cfg.log_interval,
        "precision": cfg.precision,
        "output_dir": cfg.output_dir,
        "gradient_accumulation": cfg.gradient_accumulation,
        "save_checkpoints": cfg.save_checkpoints,
        "checkpoint_interval": cfg.checkpoint_interval,
        "checkpoint_dir": cfg.checkpoint_dir,
        "resume_from": cfg.resume_from,
        "additional_args": cfg.additional_args,
    }


def _dataloader_to_dict(cfg: DataLoaderConfig) -> Dict[str, Any]:
    return {
        "batch_size": cfg.batch_size,
        "shuffle": cfg.shuffle,
        "num_workers": cfg.num_workers,
        "drop_last": cfg.drop_last,
        "pin_memory": cfg.pin_memory,
        "additional_args": cfg.additional_args,
    }


def load_yaml_config(path: str | Path) -> TrainingConfig:
    """Load a YAML configuration file into :class:`TrainingConfig`."""

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    if not isinstance(raw_cfg, Mapping):
        raise TypeError("Top-level configuration must be a mapping")

    datasets = _parse_datasets(raw_cfg.get("datasets") or raw_cfg.get("dataset"))
    dataloaders = _parse_dataloaders(raw_cfg.get("dataloaders") or raw_cfg.get("dataloader"))
    model_cfg = _parse_model(raw_cfg.get("model"))
    optimizer_cfg = _parse_component(raw_cfg.get("optimizer"), OptimizerConfig)
    scheduler_cfg = None
    if "scheduler" in raw_cfg and raw_cfg["scheduler"] is not None:
        scheduler_cfg = _parse_component(raw_cfg["scheduler"], SchedulerConfig)
    trainer_cfg = _parse_trainer(raw_cfg.get("trainer"))

    return TrainingConfig(
        datasets=datasets,
        dataloaders=dataloaders,
        model=model_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        trainer=trainer_cfg,
    )


def _ensure_mapping(section: Any, name: str) -> MutableMapping[str, Any]:
    if section is None:
        raise KeyError(f"Missing required configuration section: {name}")
    if not isinstance(section, MutableMapping):
        raise TypeError(f"Configuration section '{name}' must be a mapping")
    return section


def _parse_datasets(section: Any) -> Dict[str, DatasetConfig]:
    section = _ensure_mapping(section, "datasets")
    parsed: Dict[str, DatasetConfig] = {}
    for split, value in section.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataset configuration for split '{split}' must be a mapping")
        if "name" not in value:
            raise KeyError(f"Dataset configuration for split '{split}' requires a 'name'")
        params = value.get("params", {}) or {}
        if not isinstance(params, Mapping):
            raise TypeError(f"Dataset params for split '{split}' must be a mapping")
        parsed[split] = DatasetConfig(name=value["name"], params=dict(params))
    if not parsed:
        raise ValueError("At least one dataset split must be configured")
    return parsed


def _parse_dataloaders(section: Any) -> Dict[str, DataLoaderConfig]:
    section = _ensure_mapping(section, "dataloaders")
    parsed: Dict[str, DataLoaderConfig] = {}
    for split, value in section.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataloader configuration for split '{split}' must be a mapping")
        cfg = DataLoaderConfig(
            batch_size=int(value.get("batch_size", 32)),
            shuffle=bool(value.get("shuffle", split == "train")),
            num_workers=int(value.get("num_workers", 0)),
            drop_last=bool(value.get("drop_last", False)),
            pin_memory=bool(value.get("pin_memory", False)),
            additional_args=dict(value.get("additional_args", {}) or {}),
        )
        parsed[split] = cfg
    if not parsed:
        raise ValueError("At least one dataloader must be configured")
    return parsed


def _parse_model(section: Any) -> ModelConfig:
    section = _ensure_mapping(section, "model")
    if "backbone" not in section:
        raise KeyError("Model configuration requires a 'backbone' section")
    if "head" not in section:
        raise KeyError("Model configuration requires a 'head' section")
    backbone = _parse_component(section["backbone"], ComponentConfig)
    head = _parse_component(section["head"], ComponentConfig)
    params = dict(section.get("params", {}) or {})
    return ModelConfig(backbone=backbone, head=head, params=params)


def _parse_component(section: Any, cls: type[ComponentConfig]) -> ComponentConfig:
    section = _ensure_mapping(section, cls.__name__.lower())
    if "name" not in section:
        raise KeyError(f"{cls.__name__} configuration requires a 'name'")
    params = section.get("params", {}) or {}
    if not isinstance(params, Mapping):
        raise TypeError(f"Parameters for {cls.__name__} must be a mapping")
    return cls(name=section["name"], params=dict(params))


def _parse_trainer(section: Any) -> TrainerConfig:
    section = _ensure_mapping(section, "trainer")
    additional_args = dict(section.get("additional_args", {}) or {})
    checkpoint_interval = int(section.get("checkpoint_interval", 1))
    if checkpoint_interval <= 0:
        raise ValueError("Trainer 'checkpoint_interval' must be a positive integer")
    checkpoint_dir = section.get("checkpoint_dir")
    if checkpoint_dir is not None:
        checkpoint_dir = str(checkpoint_dir)
    resume_from = section.get("resume_from")
    if resume_from is not None:
        resume_from = str(resume_from)
    return TrainerConfig(
        max_epochs=int(section.get("max_epochs", 1)),
        device=str(section.get("device", "cpu")),
        log_interval=int(section.get("log_interval", 10)),
        precision=str(section.get("precision", "float32")),
        output_dir=str(section.get("output_dir", "outputs")),
        gradient_accumulation=int(section.get("gradient_accumulation", 1)),
        save_checkpoints=bool(section.get("save_checkpoints", True)),
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        additional_args=additional_args,
    )


__all__ = [
    "ComponentConfig",
    "DatasetConfig",
    "DataLoaderConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainerConfig",
    "TrainingConfig",
    "load_yaml_config",
]
