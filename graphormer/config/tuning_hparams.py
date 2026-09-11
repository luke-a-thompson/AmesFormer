from dataclasses import dataclass
from typing import List, Optional, Tuple
import click
from random import random
from graphormer.config.data import DataConfig
from graphormer.config.hparams import HyperparameterConfig, hyperparameters
from graphormer.config.options import (
    AttentionType,
    DatasetRegime,
    LossReductionType,
    OptimizerType,
    ResidualType,
    SchedulerType,
    NormType,
    LossFunction,
)
from optuna.trial import Trial

from graphormer.utils import create_composite_decorator, configure, resolve_device

tuning_hyperparameters = create_composite_decorator(
    hyperparameters,
    click.option(
        "-c",
        "--config",
        type=click.Path(dir_okay=False),
        is_eager=True,
        expose_value=False,
        help="Read option values from the specified config file",
        callback=configure,
        default="hparams/default_tuning_hparams.toml",
    ),
    click.option("--logdir", default="optuna_runs"),
    click.option("--max_in_degree", default=5),
    click.option("--max_out_degree", default=5),
    click.option("--max_path_distance", default=5),
    click.option("--min_b1", default=0.7),
    click.option("--max_b1", default=0.99),
    click.option("--min_b2", default=0.99),
    click.option("--max_b2", default=0.9999),
    click.option("--min_weight_decay", default=0.0),
    click.option("--max_weight_decay", default=1e-3),
    click.option("--min_eps", default=1e-10),
    click.option("--max_eps", default=1e-7),
    click.option("--min_momentum", default=0.0),
    click.option("--max_momentum", default=1.0),
    click.option("--min_max_momentum", default=0.9),
    click.option("--max_max_momentum", default=0.99),
    click.option("--min_base_momentum", default=0.1),
    click.option("--max_base_momentum", default=1.0),
    click.option("--min_dampening", default=0.0),
    click.option("--max_dampening", default=1.0),
    click.option("--min_clip_grad_norm", default=1.0),
    click.option("--max_clip_grad_norm", default=7.0),
    click.option("--min_lr_power", default=0.1),
    click.option("--max_lr_power", default=0.9),
    click.option("--min_lr_patience", default=0),
    click.option("--max_lr_patience", default=5),
    click.option("--min_lr_cooldown", default=0),
    click.option("--max_lr_cooldown", default=3),
    click.option("--min_lr_min", default=1e-7),
    click.option("--max_lr_min", default=3e-5),
    click.option("--min_lr_max", default=1e-4),
    click.option("--max_lr_max", default=1e-2),
    click.option("--min_lr_warmup", default=0),
    click.option("--max_lr_warmup", default=3),
    click.option("--min_lr_window", default=1),
    click.option("--max_lr_window", default=5),
    click.option("--min_lr_reset", default=0),
    click.option("--max_lr_reset", default=7),
    click.option("--min_lr_factor", default=0.1),
    click.option("--max_lr_factor", default=0.9),
    click.option("--min_pct_start", default=0.1),
    click.option("--max_pct_start", default=0.9),
    click.option("--min_div_factor", default=1.0),
    click.option("--max_div_factor", default=1e2),
    click.option("--min_final_div_factor", default=1.0),
    click.option("--max_final_div_factor", default=1e6),
    click.option("--study_name", default=None),
    click.option("--min_dropout", default=0.0),
    click.option("--max_dropout", default=0.5),
    click.option("--tune_size", default=0.25),
    click.option("--n_trials", default=10000),
)


@dataclass(eq=False, repr=False)
class TuningHyperparameterConfig:
    study_name: str
    random_state: int = int(random() * 1000000000.0)
    torch_device: str = 'auto'
    checkpoint_dir: str = 'optuna_models'
    n_trials: Optional[int] = None
    dataset: Optional[str] = None
    dataset_regime: Optional[DatasetRegime] = None
    node_feature_dim: Optional[int] = None
    edge_feature_dim: Optional[int] = None
    max_path_distance: Optional[int] = None
    batch_size: Optional[int] = None
    test_size: Optional[float] = None
    tune_size: Optional[float] = None
    num_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    edge_embedding_dim: Optional[int] = None
    ffn_hidden_dim: Optional[int] = None
    n_heads: Optional[int] = None
    heads_by_layer: Optional[List[int]] = None
    max_in_degree: Optional[int] = None
    max_out_degree: Optional[int] = None
    output_dim: Optional[int] = None
    max_dropout: Optional[float] = None
    min_dropout: Optional[float] = None
    norm_type: Optional[NormType] = None
    attention_type: Optional[AttentionType] = None
    residual_type: Optional[ResidualType] = None
    optimizer_type: Optional[OptimizerType] = None
    loss_reduction_type: Optional[LossReductionType] = None
    lr: Optional[float] = None
    b1: Optional[float] = None
    b2: Optional[float] = None
    max_b1: Optional[float] = None
    min_b1: Optional[float] = None
    max_b2: Optional[float] = None
    min_b2: Optional[float] = None
    weight_decay: Optional[float] = None
    max_weight_decay: Optional[float] = None
    min_weight_decay: Optional[float] = None
    eps: Optional[float] = None
    max_eps: Optional[float] = None
    min_eps: Optional[float] = None
    clip_grad_norm: float = 3.0
    max_clip_grad_norm: float = 5.0
    min_clip_grad_norm: float = 1.0
    momentum: Optional[float] = None
    max_momentum: Optional[float] = None
    min_momentum: Optional[float] = None
    dampening: Optional[float] = None
    max_dampening: Optional[float] = None
    min_dampening: Optional[float] = None
    nesterov: Optional[float] = None
    loss_function: Optional[LossFunction | tuple[LossFunction, ...]] = None
    loss_weights: Optional[Tuple[float, ...]] = None
    scheduler_type: Optional[SchedulerType] = None
    lr_power: Optional[float] = None
    lr_min: Optional[float] = None
    lr_max: Optional[float] = None
    max_lr_power: Optional[float] = None
    min_lr_power: Optional[float] = None
    lr_patience: Optional[int] = None
    max_lr_patience: Optional[int] = None
    min_lr_patience: Optional[int] = None
    lr_cooldown: Optional[int] = None
    max_lr_cooldown: Optional[int] = None
    min_lr_cooldown: Optional[int] = None
    max_lr_min: Optional[float] = None
    min_lr_min: Optional[float] = None
    max_lr_max: Optional[float] = None
    min_lr_max: Optional[float] = None
    lr_warmup: Optional[int] = None
    lr_smooth: Optional[bool] = None
    max_lr_warmup: Optional[int] = None
    min_lr_warmup: Optional[int] = None
    lr_window: Optional[int] = None
    max_lr_window: Optional[int] = None
    min_lr_window: Optional[int] = None
    lr_reset: Optional[float] = None
    max_lr_reset: Optional[int] = None
    min_lr_reset: Optional[int] = None
    lr_factor: Optional[float] = None
    max_lr_factor: Optional[float] = None
    min_lr_factor: Optional[float] = None
    pct_start: Optional[float] = None
    min_pct_start: Optional[float] = None
    max_pct_start: Optional[float] = None
    div_factor: Optional[float] = None
    min_div_factor: Optional[float] = None
    max_div_factor: Optional[float] = None
    final_div_factor: Optional[float] = None
    min_final_div_factor: Optional[float] = None
    max_final_div_factor: Optional[float] = None
    cycle_momentum: Optional[bool] = None
    three_phase: Optional[bool] = None
    min_max_momentum: Optional[float] = None
    max_max_momentum: Optional[float] = None
    base_momentum: Optional[float | List[float]] = None
    min_base_momentum: Optional[float] = None
    max_base_momentum: Optional[float] = None
    anneal_strategy: Optional[str] = None
    last_effective_batch_num: Optional[int] = None
    logdir: Optional[str] = None
    epochs: int = 40
    accumulation_steps: Optional[int] = None

    def __post_init__(self):
        self.torch_device = resolve_device(self.torch_device)

    def create_hyperparameters(self, trial: Trial) -> HyperparameterConfig:
        for field in (
            'batch_size', 'min_dropout', 'max_dropout', 'lr', 'min_b1', 'max_b1',
            'min_b2', 'max_b2', 'min_weight_decay', 'max_weight_decay', 'min_eps', 'max_eps',
            'min_clip_grad_norm', 'max_clip_grad_norm', 'min_momentum', 'max_momentum', 'min_dampening', 'max_dampening',
            'min_lr_power', 'max_lr_power', 'min_lr_patience', 'max_lr_patience', 'min_lr_cooldown', 'max_lr_cooldown',
            'min_lr_min', 'max_lr_min', 'min_lr_max', 'max_lr_max', 'min_lr_warmup', 'max_lr_warmup',
            'min_lr_window', 'max_lr_window', 'min_lr_reset', 'max_lr_reset', 'min_lr_factor', 'max_lr_factor',
            'accumulation_steps', 'tune_size', 'min_pct_start', 'max_pct_start', 'min_div_factor', 'max_div_factor',
            'min_final_div_factor', 'max_final_div_factor', 'min_base_momentum', 'max_base_momentum', 'min_max_momentum', 'max_max_momentum',
        ):
            if getattr(self, field) is None:
                label = "min_lr" if field == "lr" else field
                raise AttributeError(f"{label} not defined for TuningHyperparameterConfig")

        if self.residual_type is None:
            self.residual_type = ResidualType(
                trial.suggest_categorical("residual_type", [ResidualType.PRENORM, ResidualType.REZERO])
            )
        if self.optimizer_type is None:
            self.optimizer_type = OptimizerType(
                trial.suggest_categorical("optimizer_type", [OptimizerType.SGD, OptimizerType.ADAMW])
            )
        if self.scheduler_type is None:
            self.scheduler_type = SchedulerType(
                trial.suggest_categorical(
                    "scheduler_type",
                    [
                        SchedulerType.GREEDY,
                        SchedulerType.POLYNOMIAL,
                        SchedulerType.PLATEAU,
                        SchedulerType.ONE_CYCLE,
                        SchedulerType.FIXED,
                    ],
                )
            )
        loss_reduction_type = self.loss_reduction_type
        if loss_reduction_type is None:
            loss_reduction_type = LossReductionType(
                trial.suggest_categorical("loss_reduction", [LossReductionType.MEAN, LossReductionType.SUM])
            )

        if self.norm_type is None:
            self.norm_type = NormType(
                trial.suggest_categorical("norm_type", [NormType.LAYER, NormType.MAX, NormType.RMS, NormType.CRMS])
            )

        match self.optimizer_type:
            case OptimizerType.SGD:
                if self.momentum is None:
                    self.momentum = trial.suggest_float("momentum", self.min_momentum, self.max_momentum)

                if self.dampening is None:
                    self.dampening = trial.suggest_float("dampening", self.min_dampening, self.max_dampening)

                if self.nesterov is None:
                    self.nesterov = trial.suggest_categorical("nesterov", [True, False])
            case OptimizerType.ADAMW:
                if self.b1 is None:
                    self.b1 = trial.suggest_float("b1", self.min_b1, self.max_b1)
                if self.b2 is None:
                    self.b2 = trial.suggest_float("b2", self.min_b2, self.max_b2)
                if self.eps is None:
                    self.eps = trial.suggest_float("eps", self.min_eps, self.max_eps)

        match self.scheduler_type:
            case SchedulerType.GREEDY:
                self.lr_smooth = trial.suggest_categorical("lr_smooth", [True, False])
                self.lr_warmup = trial.suggest_int("lr_warmup", self.min_lr_warmup, self.max_lr_warmup)
                self.lr_cooldown = trial.suggest_int("lr_cooldown", self.min_lr_cooldown, self.max_lr_cooldown)
                self.lr_window = trial.suggest_int("lr_window", self.min_lr_window, self.max_lr_window)
                self.lr_patience = trial.suggest_int("lr_patience", self.min_lr_patience, self.max_lr_patience)
                self.lr_reset = trial.suggest_int("lr_reset", self.min_lr_reset, self.max_lr_reset)
                self.lr_max = trial.suggest_float("lr_max", self.min_lr_max, self.max_lr_max)
                self.lr_min = trial.suggest_float("lr_min", self.min_lr_min, self.max_lr_min)
                self.lr_factor = trial.suggest_float("lr_factor", self.min_lr_factor, self.max_lr_factor)
            case SchedulerType.PLATEAU:
                self.lr_min = trial.suggest_float("lr_min", self.min_lr_min, self.max_lr_min)
                self.lr_factor = trial.suggest_float("lr_factor", self.min_lr_factor, self.max_lr_factor)
                self.lr_patience = trial.suggest_int("lr_patience", self.min_lr_patience, self.max_lr_patience)
                self.lr_cooldown = trial.suggest_int("lr_cooldown", self.min_lr_cooldown, self.max_lr_cooldown)
            case SchedulerType.POLYNOMIAL:
                self.lr_power = trial.suggest_float("lr_power", self.min_lr_power, self.max_lr_power)
            case SchedulerType.ONE_CYCLE:
                self.pct_start = trial.suggest_float("pct_start", self.min_pct_start, self.max_pct_start)
                self.div_factor = trial.suggest_float("div_factor", self.min_div_factor, self.max_div_factor)
                self.final_div_factor = trial.suggest_float(
                    "final_div_factor", self.min_final_div_factor, self.max_final_div_factor
                )
                if self.cycle_momentum is None:
                    self.cycle_momentum = trial.suggest_categorical("cycle_momentum", [True, False])
                if self.three_phase is None:
                    self.three_phase = trial.suggest_categorical("three_phase", [True, False])
                self.max_momentum = trial.suggest_float("max_momentum", self.min_max_momentum, self.max_max_momentum)
                self.base_momentum = trial.suggest_float(
                    "base_momentum", self.min_base_momentum, self.max_base_momentum
                )
                if self.anneal_strategy is None:
                    self.anneal_strategy = trial.suggest_categorical("anneal_strategy", ["cos", "linear"])

        if self.weight_decay is None:
            self.weight_decay = trial.suggest_float("weight_decay", self.min_weight_decay, self.max_weight_decay)

        if self.clip_grad_norm is None:
            self.clip_grad_norm = trial.suggest_float(
                "clip_grad_norm", self.min_clip_grad_norm, self.max_clip_grad_norm
            )

        return HyperparameterConfig(
            dataset=self.dataset,
            dataset_regime=self.dataset_regime,
            random_state=self.random_state,
            torch_device=self.torch_device,
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            test_size=self.test_size,
            tune_size=self.tune_size,
            logdir=self.logdir,
            start_epoch=0,
            max_path_distance=self.max_path_distance,
            # Model Parameters
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            edge_embedding_dim=self.edge_embedding_dim,
            ffn_hidden_dim=self.ffn_hidden_dim,
            n_heads=self.n_heads,
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            output_dim=self.output_dim,
            dropout=trial.suggest_float("dropout", self.min_dropout, self.max_dropout),
            batch_size=self.batch_size,
            norm_type=self.norm_type,
            attention_type=self.attention_type,
            heads_by_layer=self.heads_by_layer,
            residual_type=self.residual_type,
            # Optimizer Parameters
            optimizer_type=self.optimizer_type,
            lr=self.lr,
            b1=self.b1,
            b2=self.b2,
            weight_decay=self.weight_decay,
            eps=self.eps,
            clip_grad_norm=self.clip_grad_norm,
            momentum=self.momentum,
            dampening=self.dampening,
            nesterov=self.nesterov,
            loss_function=self.loss_function,
            loss_weights=self.loss_weights,
            # Scheduler Parameters
            scheduler_type=self.scheduler_type,
            last_effective_batch_num=self.last_effective_batch_num,
            lr_power=self.lr_power,
            lr_patience=self.lr_patience,
            lr_cooldown=self.lr_cooldown,
            lr_min=self.lr_min,
            lr_max=self.lr_max,
            lr_warmup=self.lr_warmup,
            lr_window=self.lr_window,
            lr_reset=self.lr_reset,
            lr_factor=self.lr_factor,
            lr_smooth=self.lr_smooth,
            pct_start=self.pct_start,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            cycle_momentum=self.cycle_momentum,
            three_phase=self.three_phase,
            max_momentum=self.max_momentum,
            base_momentum=self.base_momentum,
            anneal_strategy=self.anneal_strategy,
            # Training Parameters
            epochs=40,
            accumulation_steps=self.accumulation_steps,
            loss_reduction=loss_reduction_type,
            checkpoint_dir=self.checkpoint_dir,
            checkpt_save_interval=1000,
            write_to_disk=False,
        )

    def data_config(self) -> DataConfig:
        if self.dataset is None:
            raise AttributeError("dataset not defined for DataConfig")
        if self.dataset_regime is None:
            raise AttributeError("dataset_regime not defined for DataConfig")
        if self.batch_size is None:
            raise AttributeError("batch_size not defined for DataConfig")
        if self.max_path_distance is None:
            raise AttributeError("max_path_distance not defined for DataConfig")

        config = DataConfig(self.dataset, self.dataset_regime, self.batch_size, self.max_path_distance)

        if self.test_size is not None:
            config = config.with_test_size(self.test_size)
        return config
