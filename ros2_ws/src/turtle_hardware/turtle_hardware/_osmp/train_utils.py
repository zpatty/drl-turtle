__all__ = ["train", "test", "train_identity_encoder"]
import copy
import time

import matplotlib.pyplot as plt
from osmp.models import dynamics_models
from scipy.ndimage import label
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional

from osmp.integration import rollout_trajectories_euler
from osmp.models.dynamics_models import HopfBifurcationDynamics, RNNDynamics


def train(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    train_dataset: Dataset,
    batch_size: Optional[int] = None,
    oracle_ts: Optional[torch.Tensor] = None,
    oracle_x_ts: Optional[torch.Tensor] = None,
    num_rollouts_per_batch: int = 0,
    num_rollout_steps: int = 100,
    n_epochs: int = 1000,
    n_warmup_epochs: int = 10,
    n_cosine_annealing_epochs: int = 300,
    shuffle: bool = True,
    loss_x_fn: Callable = nn.MSELoss(reduction="mean"),
    loss_x_d_fn: Callable = nn.MSELoss(reduction="mean"),
    loss_y_fn: Optional[Callable] = None,
    loss_x_weight: float = 1.0,
    loss_x_d_weight: float = 1.0,
    loss_y_weight: float = 1.0,
    loss_z_inter_weight: float = 1.0,
    clip_gradient: bool = True,
    clip_value_grad=0.1,
    loss_clip: float = 1e3,
    stop_threshold: float = float("inf"),
    return_best_model: bool = True,
    log_freq: int = 5,
    logger=None,
    on_epoch_end_callback: Optional[Callable] = None,
):
    """
    train the torch model with the given parameters
    :param model (torch.nn.Module): the model to be trained
    :param loss_fn (callable): loss = loss_fn(y_pred, y_target)
    :param opt (torch.optim): optimizer
    :param train_dataset (torch.utils.data.Dataset): training dataset
    :param batch_size (int): size of minibatch, if None, train in batch
    :param oracle_ts Optional[torch.Tensor]: time points for rollout of the trajectory.
        If None, no rollout is performed and no loss on task-space position predictions is applied.
    :param oracle_x_ts Optional[torch.Tensor]: task-space positions for the rollout of the trajectory.
    :param num_rollouts_per_batch (int): number of rollouts per batch
    :param n_epochs (int): number of epochs
    :param n_warmup_epochs (int): number of epochs for warm-up
    :param learning_rate_schedule_type (str): type of learning rate schedule: constant or cosine_annealing
    :param shuffle (bool): whether the dataset is reshuffled at every epoch
    :param loss_y_fn (callable): loss = loss_y_fn(y)
    :param loss_x_d_weight (float): weight for the loss on the task-space velocity predictions
    :param loss_x_weight (float): weight for the loss on the task-space position predictions
    :param loss_y_weight (float): weight for the loss on the latent-space encodings
    :param clip_gradient (bool): whether the gradients are clipped
    :param clip_value_grad (float): the threshold for gradient clipping
    :param loss_clip (float): the threshold for loss clipping
    :param stop_threshold (float): if the loss does not decrease for stop_threshold epochs, stop training
    :param return_best_model (bool): whether to return the best model or the last model
    :param log_freq (int): the epoch frequency for printing loss and saving results on tensorboard
    :param logger: the tensorboard logger
    :param on_epoch_end_callback: the callback function to be called at the end of each epoch
    :return: None
    """

    # if batch_size is None, train in batch
    n_samples = len(train_dataset)
    if batch_size is None:
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=n_samples, shuffle=shuffle
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
        )

    # move tensors to device
    if oracle_ts is not None:
        oracle_ts = oracle_ts.to(model.device)
        oracle_ts_indices = torch.arange(0, oracle_ts.size(0), device=model.device)
    if oracle_x_ts is not None:
        oracle_x_ts = oracle_x_ts.to(model.device)

    assert n_warmup_epochs >= 0
    assert n_cosine_annealing_epochs >= 0
    assert n_warmup_epochs + n_cosine_annealing_epochs <= n_epochs
    scheduler = LinearLR(
        opt,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=min(n_epochs, n_warmup_epochs),
    )
    schedulers = [scheduler]
    milestones = []
    num_remaining_epochs = n_epochs - n_warmup_epochs
    if n_epochs - n_warmup_epochs > 0:
        milestones += [n_warmup_epochs]
        if n_cosine_annealing_epochs < num_remaining_epochs:
            scheduler = ConstantLR(
                opt, factor=1.0, total_iters=n_cosine_annealing_epochs
            )
            schedulers += [scheduler]
            milestones += [n_warmup_epochs + n_cosine_annealing_epochs]
            num_remaining_epochs -= n_cosine_annealing_epochs

        scheduler = CosineAnnealingLR(opt, T_max=num_remaining_epochs)
        schedulers += [scheduler]
        num_remaining_epochs = 0
    scheduler = SequentialLR(opt, schedulers=schedulers, milestones=milestones)

    # record time elasped
    wallclock_start = time.time()

    if loss_x_d_fn.reduction == "mean":
        mean_flag = True
    else:
        mean_flag = False

    best_train_loss = float("inf")
    best_train_epoch = 0
    best_model = model

    # train the model
    model.train()
    for epoch in range(n_epochs):
        # iterate over minibatches
        train_loss = 0.0
        for batch_idx, (z_bt, x_bt, x_d_target_bt) in enumerate(train_loader):
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)

            if loss_x_d_weight > 0.0 or loss_y_weight > 0.0:
                # move data to device
                z_bt = z_bt.to(model.device)
                if isinstance(x_bt, torch.Tensor):
                    x_bt = x_bt.to(model.device)
                elif isinstance(x_bt, dict):
                    x_bt = {k: v.to(model.device) for k, v in x_bt.items()}
                else:
                    raise NotImplementedError
                if isinstance(x_d_target_bt, torch.Tensor):
                    x_d_target_bt = x_d_target_bt.to(model.device)
                elif isinstance(x_d_target_bt, dict):
                    x_d_target_bt = {
                        k: v.to(model.device) for k, v in x_d_target_bt.items()
                    }
                else:
                    raise NotImplementedError

                # forward pass
                model_kwargs = dict()
                if model.condition_encoding:
                    model_kwargs["z"] = z_bt

                x_d_pred, aux = model(x_bt, **model_kwargs)

                if loss_x_d_fn is not None and loss_x_d_weight > 0.0:
                    # compute loss
                    loss_x_d = loss_x_d_fn(x_d_pred, x_d_target_bt)
                    loss = loss + loss_x_d_weight * loss_x_d

                if loss_y_fn is not None and loss_y_weight > 0.0:
                    # compute the loss between y and the closest point on the limit cycle
                    loss_y = loss_y_fn(aux["y"])
                    loss = loss + loss_y_weight * loss_y

            if num_rollouts_per_batch > 0 and loss_x_weight > 0.0:
                # randomly sample num_rollouts_per_batch time indices
                start_time_indices = torch.randint(
                    low=0,
                    high=oracle_ts.size(0) - num_rollout_steps,
                    size=(num_rollouts_per_batch,),
                    device=model.device,
                )
                end_time_indices = start_time_indices + num_rollout_steps + 1

                # construct time batch of shape (num_rollouts_per_batch, num_rollout_steps)
                time_selector = (oracle_ts_indices >= start_time_indices[:, None]) & (
                    oracle_ts_indices < end_time_indices[:, None]
                )
                ts_bt = torch.repeat_interleave(
                    oracle_ts[None, ...], repeats=num_rollouts_per_batch, dim=0
                )[time_selector].reshape(num_rollouts_per_batch, -1)

                # select initial conditions of shape (batch_size, state_dim)
                x0_bt = oracle_x_ts[start_time_indices, :]

                # target state trajectory
                x_target_bt = torch.repeat_interleave(
                    oracle_x_ts[None, ...], repeats=num_rollouts_per_batch, dim=0
                )[time_selector].reshape(
                    num_rollouts_per_batch, num_rollout_steps + 1, -1
                )

                if type(model.dynamics_model) is RNNDynamics:
                    h0_bt = torch.zeros(
                        num_rollouts_per_batch,
                        model.dynamics_model.total_hidden_state_dim,
                        device=model.device,
                    )
                else:
                    h0_bt = None

                # perform the rollout
                rollout_pred_ts = rollout_trajectories_euler(
                    model, x0_bt, ts_bt, h0=h0_bt
                )
                x_pred_bt = rollout_pred_ts["y_ts"]

                # compute the loss on the task-space position predictions
                loss_x = loss_x_fn(x_pred_bt, x_target_bt)
                loss = loss + loss_x_weight * loss_x

            if (
                loss_z_inter_weight > 0.0
                and model.condition_encoding
                and type(model.dynamics_model) is HopfBifurcationDynamics
            ):
                num_polar_angles = 10
                num_condition_interpolations = 10

                # randomly sample num_polar_angles angles
                varphis = (
                    torch.rand(num_polar_angles, device=model.device) * 2 * torch.pi
                )

                # get the radius of the limit cycle
                R = model.dynamics_model.get_limit_cycle_radius()

                # compute the Cartesian coordinates
                y = torch.cat(
                    [
                        R * torch.cos(varphis)[..., None],
                        R * torch.sin(varphis)[..., None],
                        torch.zeros(
                            (varphis.size(0), model.n_dim_x - 2), device=model.device
                        ),
                    ],
                    dim=-1,
                )
                # repeat the condition for num_condition_interpolations
                y_repeat_inter = torch.repeat_interleave(
                    y, repeats=num_condition_interpolations, dim=0
                )
                # sample num_condition_interpolations random z
                z_min, z_max = z_bt.min(), z_bt.max()
                z_inter = (
                    torch.rand(
                        (num_polar_angles * num_condition_interpolations, z_bt.size(1)),
                        device=model.device,
                    )
                    * (z_max - z_min)
                    + z_min
                )
                # map the samples into task space
                x_inter = model.taskmap_model.predict(
                    y_repeat_inter, z=z_inter, mode=-1
                )
                # map the min and max z into task space for each y sample
                z_repeat_min_max = torch.stack([z_min, z_max], dim=0)[:, None].repeat(
                    num_polar_angles, 1
                )
                y_repeat_min_max = torch.repeat_interleave(y, repeats=2, dim=0)
                x_min_max = model.taskmap_model.predict(
                    y_repeat_min_max, z=z_repeat_min_max, mode=-1
                )
                x_min = x_min_max[::2]
                x_max = x_min_max[1::2]
                x_min_repeat_inter = torch.repeat_interleave(
                    x_min, repeats=num_condition_interpolations, dim=0
                )
                x_max_repeat_inter = torch.repeat_interleave(
                    x_max, repeats=num_condition_interpolations, dim=0
                )
                # interpolate the min and max x to get the target x
                x_inter_targets = (
                    x_min_repeat_inter
                    + (x_max_repeat_inter - x_min_repeat_inter) * z_inter
                )
                # compute the loss
                loss_z_inter = torch.mean((x_inter - x_inter_targets).pow(2))
                loss = loss + loss_z_inter_weight * loss_z_inter

                if epoch == n_epochs - 1 and batch_idx == 0:
                    fig = plt.figure()
                    ax = plt.gca()
                    # plot the oracle
                    plt.plot(
                        x_bt[:, 0].detach().cpu().numpy(),
                        x_bt[:, 1].detach().cpu().numpy(),
                        linestyle="None",
                        marker=".",
                        markersize=3,
                        color="k",
                        label="Oracles",
                    )
                    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                    for i in range(num_polar_angles):
                        xi_min_max_plt = (
                            torch.stack([x_min[i], x_max[i]], dim=0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # line connecting min and max
                        plt.plot(
                            xi_min_max_plt[:, 0],
                            xi_min_max_plt[:, 1],
                            linewidth=2.0,
                            color=color_cycle[i],
                            label=f"min-max line {i + 1}",
                        )
                        # plot the min and max
                        plt.plot(
                            xi_min_max_plt[0, 0],
                            xi_min_max_plt[0, 1],
                            linestyle="None",
                            marker="x",
                            markersize=7,
                            color=color_cycle[i],
                            label=f"min {i + 1}",
                        )
                        plt.plot(
                            xi_min_max_plt[1, 0],
                            xi_min_max_plt[1, 1],
                            linestyle="None",
                            marker="*",
                            markersize=7,
                            color=color_cycle[i],
                            label=f"max {i + 1}",
                        )
                    for i in range(x_inter_targets.size(0)):
                        plt.plot(
                            x_inter_targets[i, 0].detach().cpu().numpy(),
                            x_inter_targets[i, 1].detach().cpu().numpy(),
                            linestyle="None",
                            marker="P",
                            # label="target"
                        )
                    plt.gca().set_prop_cycle(None)
                    for i in range(x_inter_targets.size(0)):
                        plt.plot(
                            x_inter[i, 0].detach().cpu().numpy(),
                            x_inter[i, 1].detach().cpu().numpy(),
                            linestyle="None",
                            marker="o",
                            # label="predictions"
                        )
                    plt.xlabel(r"$x_1$ [m]")
                    plt.ylabel(r"$x_2$ [m]")
                    plt.legend()
                    plt.grid(True)
                    plt.box(True)
                    plt.show()

            if loss_x_d_weight > 0.0:
                train_loss += loss_x_d.detach().cpu().item()
            elif loss_x_weight > 0.0:
                train_loss += loss_x.detach().cpu().item()
            elif loss_y_weight > 0.0:
                train_loss += loss_y.detach().cpu().item()
            elif loss_z_inter_weight > 0.0:
                train_loss += loss_z_inter.detach().cpu().item()
            else:
                train_loss += loss.detach().cpu().item()

            if loss > loss_clip:
                print("loss too large, skip")
                continue

            # backward pass
            opt.zero_grad()
            loss.backward()

            # clip gradient based on norm
            if clip_gradient:
                # torch.nn.utils.clip_grad_value_(
                #     model.parameters(),
                #     clip_value_grad
                # )
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value_grad)
            # update parameters
            opt.step()

        scheduler.step()
        # print("Current learning rate: ", opt.param_groups[-1]['lr'])

        if mean_flag:  # fix for taking mean over all data instead of mini batch!
            train_loss = float(batch_size) / float(n_samples) * train_loss

        if epoch - best_train_epoch >= stop_threshold:
            break

        if train_loss < best_train_loss:
            best_train_epoch = epoch
            best_train_loss = train_loss
            best_model = copy.deepcopy(model)

        # report loss in command line and tensorboard every log_freq epochs
        if epoch % log_freq == (log_freq - 1):
            print(
                "    Epoch [{}/{}]: current loss is {}, time elapsed {} second".format(
                    epoch + 1, n_epochs, train_loss, time.time() - wallclock_start
                )
            )

            if logger is not None:
                info = {"Training Loss": train_loss}

                # log scalar values (scalar summary)
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                # log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace(".", "/")
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                    logger.histo_summary(
                        tag + "/grad", value.grad.data.cpu().numpy(), epoch + 1
                    )

        if on_epoch_end_callback is not None:
            on_epoch_end_callback(epoch, train_loss)

    if return_best_model:
        return best_model, best_train_loss
    else:
        return model, train_loss


def test(model, loss_fn, x_test, y_test):
    """
    test the torch model
    :param: model (torch.nn.Model): the trained torch model
    :param: loss_fn (callable): loss=loss_fn(y_pred, y_target)
    :param: x_test (torch.Tensor): test input
    :param: y_test (torch.Tensor): test target output
    :return: loss (float): loss over the test set
    """
    model.eval()

    # move data to device
    if isinstance(x_test, torch.Tensor):
        x_batch = x_test.to(model.device)
    elif isinstance(x_test, dict):
        x_batch = {k: v.to(model.device) for k, v in x_test.items()}
    else:
        raise NotImplementedError
    if isinstance(y_test, torch.Tensor):
        y_batch = y_test.to(model.device)
    elif isinstance(y_test, dict):
        y_batch = {k: v.to(model.device) for k, v in y_test.items()}
    else:
        raise NotImplementedError

    y_pred, _ = model(x_test)

    return loss_fn(y_pred, y_test).item()


def train_identity_encoder(
    encoder: nn.Module,
    x_lims: torch.Tensor,
    batch_size: int = 64,
    num_iterations: int = 1000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Train an identity encoder that maps the input to itself.
    Arguments:
        encoder: the encoder to be trained to be an identity function mapping the input to itself.
        x_lims: the limits of the input space as a tensor of shape (n_dims, 2).
        batch_size: the batch size for training.
        num_iterations: the number of iterations for training.
        lr: the learning rate for training
        device: the device to use for training.
    """
    num_dims = x_lims.size(0)
    print(f"Training an identity encoder for {num_dims} dimensions.")

    # move everything to the device
    encoder = encoder.to(device)
    x_lims = x_lims.to(device)

    # define the loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # define the optimizer
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    # train the encoder
    encoder.train()
    for it in range(num_iterations):
        # generate random data
        x = torch.rand(
            (batch_size, num_dims), requires_grad=True, device=device
        ).float()
        x = x * (x_lims[:, 1] - x_lims[:, 0]) + x_lims[:, 0]

        # forward pass
        y_pred, _ = encoder(x)

        # compute loss
        loss = loss_fn(y_pred, x)

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 100 == 0:
            print(f"Iteration {it}: loss = {loss.item()}")

    return encoder
