"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python aliegnn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""
import gc

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union
import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
    # For different version of pytorch-ignite
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from aliegnn import models
from aliegnn.config import TrainingConfig
#from aliegnn.models.alignn_egnn import ALIEGNN
from aliegnn.models.alignn_egnn import ALIEGNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os

# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    # pc = pk.load(open("pca.pkl", "rb"))
    # y_pred = torch.tensor(pc.transform(y_pred), device=device)
    # y = torch.tensor(pc.transform(y), device=device)

    # y_pred = torch.tensor(pca_sc.inverse_transform(y_pred),device=device)
    # y = torch.tensor(pca_sc.inverse_transform(y),device=device)
    # print (y.shape,y_pred.shape)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    # checkpoint_dir: Path = Path("./"),
    train_val_test_loaders=[],
    # log_tensorboard: bool = False,
    cv = 0, # cross validation fold index
    predict_mat: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to aliegnn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    print(config)
    if type(config) is dict:
        try:
            print(config)
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
    import os

    # set up data loaders
    train_loader = train_val_test_loaders[0]
    val_loader = train_val_test_loaders[1]
    test_loader = train_val_test_loaders[2]
    prepare_batch = train_val_test_loaders[3]

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = False
    aliegnn_models = {
        "alignn",
    }
    if config.model.alignn_layers > 0:
        line_graph = True
    # print ('output_dir train', config.output_dir)
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        "alignn": ALIEGNN,
        #"alie3nn": ALiE3NN,
    }
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model

    net.to(device)
    if config.distributed:
        import torch.distributed as dist
        import os

        def setup(rank, world_size):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            dist.destroy_process_group()

        setup(2, 2)
        # local_rank = 0
        # net=torch.nn.parallel.DataParallel(net
        # ,device_ids=[local_rank, ],output_device=local_rank)
        net = torch.nn.parallel.DistributedDataParallel(
            net
        )  # ,device_ids=[local_rank, ],output_device=local_rank)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[config.criterion]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
    if config.model.output_features > 1 and config.standard_scalar_and_pca:
        # metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
        metrics = {
            "loss": Loss(
                criterion, output_transform=make_standard_scalar_and_pca
            ),
            "mae": MeanAbsoluteError(
                output_transform=make_standard_scalar_and_pca
            ),
        }

    if classification:
        criterion = nn.NLLLoss()

        metrics = {
            "accuracy": Accuracy(
                output_transform=thresholded_output_transform
            ),
            "precision": Precision(
                output_transform=thresholded_output_transform
            ),
            "recall": Recall(output_transform=thresholded_output_transform),
            "rocauc": ROC_AUC(output_transform=activated_output_transform),
            "roccurve": RocCurve(output_transform=activated_output_transform),
            "confmat": ConfusionMatrix(
                output_transform=thresholded_output_transform, num_classes=2
            ),
        }
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
        # output_transform=make_standard_scalar_and_pca,
    )

    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
        # output_transform=make_standard_scalar_and_pca,
    )

    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
        # output_transform=make_standard_scalar_and_pca,
    )

    # for regression, we want to save the model with the lowest MAE
    best_model_saver = None

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        # Add cv prefix to checkpoint files
        prefix = f"cv_{cv}_" if cv != 0 else ""
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            filename_prefix=prefix,
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

        # Save the best model based on validation MAE
        if not classification:

            def score_function(engine):
                return -engine.state.metrics["mae"]

            best_model_saver = Checkpoint(
                {"model": net},
                DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
                filename_prefix=f"best_cv_{cv}" if cv != 0 else "best",
                n_saved=1,
                score_function=score_function,
                score_name="val_mae",
                global_step_transform=global_step_from_engine(trainer),
            )
            evaluator.add_event_handler(Events.COMPLETED, best_model_saver)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        
        # Only run validation if we have validation data (predict_mat=False)
        if not predict_mat and val_loader is not None:
            evaluator.run(val_loader)
            vmetrics = evaluator.state.metrics
        else:
            # For predict_mat=True, use training metrics as validation metrics
            vmetrics = train_evaluator.state.metrics

        tmetrics = train_evaluator.state.metrics
        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            if metric == "roccurve":
                tm = [k.tolist() for k in tm]
                vm = [k.tolist() for k in vm]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)

        # for metric in metrics.keys():
        #    history["train"][metric].append(tmetrics[metric])
        #    history["validation"][metric].append(vmetrics[metric])

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            if not classification:
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
                if not predict_mat:
                    pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                if not predict_mat:
                    pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")

    # Only use early stopping when we have validation data (predict_mat=False)
    if config.n_early_stopping is not None and config.n_early_stopping > 0 and not predict_mat:
        if classification:
            my_metrics = "accuracy"
            def default_score_fn(engine):
                score = engine.state.metrics[my_metrics]
                return score
        else:
            my_metrics = "mae"
        # early stopping handler
            def default_score_fn(engine):
                score = engine.state.metrics[my_metrics]
                return -score

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=default_score_fn,
            trainer=trainer,
            min_delta=config.min_delta,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)
    # optionally log results to tensorboard
    if config.log_tensorboard:

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    # Load the best model for evaluation
    if best_model_saver and not predict_mat:
        best_checkpoint_path = best_model_saver.last_checkpoint
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            print(f"Loading best model for evaluation: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            Checkpoint.load_objects(to_load={"model": net}, checkpoint=checkpoint)
        else:
            print(
                "Best model checkpoint not found. Using the final model for evaluation."
            )
    elif predict_mat:
        print("Using final trained model (predict_mat=True, no validation-based selection)")
        # Save the final model as best model for prediction use
        final_model_path = os.path.join(checkpoint_dir, f"best_cv_{cv}_model.pt" if cv != 0 else "best_model.pt")
        torch.save({"model": net.state_dict()}, final_model_path)
        print(f"Final model saved as: {final_model_path}")

    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()
    if config.write_predictions and classification:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

        print("predictions", predictions)
        print("targets", targets)
        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (
        config.write_predictions
        and not classification
        and config.model.output_features > 1
    ):
        net.eval()
        mem = []
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = id
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
                targets.append(target)
                predictions.append(out_data)
        dumpjson(
            filename=os.path.join(
                config.output_dir, f"multi_out_predictions_cv_{cv}.json"
            ),
            data=mem,
        )
        from sklearn.metrics import mean_absolute_error, r2_score
        MAE = mean_absolute_error(np.array(targets), np.array(predictions))
        r2 =  r2_score(
                np.array(targets), np.array(predictions)
            )
        print(
            "Test MAE:",
            MAE,
            "\n"
            "Test R2:",
            r2,
        )
        f = open(
            os.path.join(
                config.output_dir, "prediction_metrics_results_cv.csv"
            ),
            "a",
        )
        f.write("%s, %6f, %6f\n" % (f"cv_{cv}", MAE, r2))
        f.close()
    if (
        config.write_predictions
        and not classification
        and config.model.output_features == 1
    ):
        net.eval()
        f = open(
            os.path.join(config.output_dir, f"prediction_results_test_set_cv_{cv}.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(
                        open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                    )
                    out_data = sc.transform(np.array(out_data).reshape(-1, 1))
                    [0][0]
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]
                f.write("%s, %6f, %6f\n" % (id, target, out_data))
                targets.append(target)
                predictions.append(out_data)
        f.close()
        from sklearn.metrics import mean_absolute_error, r2_score
        MAE = mean_absolute_error(np.array(targets), np.array(predictions))
        r2 =  r2_score(
                np.array(targets), np.array(predictions)
            )
        print(
            "Test MAE:",
            MAE,
            "\n"
            "Test R2:",
            r2,
        )
        f = open(
            os.path.join(
                config.output_dir, "prediction_metrics_results_cv.csv"
            ),
            "a",
        )
        f.write("%s, %6f, %6f\n" % (f"cv_{cv}", MAE, r2))
        f.close()

        net.eval()
        # net.eval()
        f = open(
            os.path.join(
                config.output_dir, f"prediction_results_train_set_cv_{cv}.csv"
            ),
            "w",
        )
        f.write("target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = train_loader.dataset.ids  
            for dat, id in zip(train_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                # out_data = net([g.to(device), lg.to(device)])["out"]
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(
                        open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                    )
                    out_data = sc.transform(
                        np.array(out_data).reshape(-1, 1)
                    )[0][0]
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                   target = target[0]
                # if len(out_data) == 1:
                #    out_data = out_data[0]
                for ii, jj in zip(target, out_data):
                    f.write("%6f, %6f\n" % (ii, jj))
                    targets.append(ii)
                    predictions.append(jj)
        f.close()

        if config.store_outputs and not classification:
            x = []
            y = []
            for i in history["EOS"]:
                x.append(i[0].cpu().numpy().tolist())
                y.append(i[1].cpu().numpy().tolist())
            x = np.array(x, dtype="float").flatten()
            y = np.array(y, dtype="float").flatten()
            f = open(
                os.path.join(
                    config.output_dir, f"prediction_results_train_set_cv_{cv}.csv"
                ),
                "w",
            )
            # TODO: Add IDs
            f.write("target,prediction\n")
            for i, j in zip(x, y):
                f.write("%6f, %6f\n" % (j, i))
                line = str(i) + "," + str(j) + "\n"
                f.write(line)
            f.close()

    # TODO: Fix IDs for train loader
    """
    if config.write_train_predictions:
        net.eval()
        f = open("train_prediction_results.csv", "w")
        f.write("id,target,prediction\n")
        with torch.no_grad():
            ids = train_loader.dataset.dataset.ids[
                train_loader.dataset.indices
            ]
            print("lens", len(ids), len(train_loader.dataset.dataset))
            x = []
            y = []

            for dat, id in zip(train_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                for i, j in zip(out_data, target):
                    x.append(i)
                    y.append(j)
            for i, j, k in zip(ids, x, y):
                f.write("%s, %6f, %6f\n" % (i, j, k))
        f.close()

    """
    del net
    del optimizer
    del trainer
    del evaluator
    del train_loader
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
