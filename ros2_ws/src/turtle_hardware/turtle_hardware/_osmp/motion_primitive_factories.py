__all__ = [
    "orbitally_stable_motion_primitive_factory",
]
import numpy as np
from os import PathLike
from pathlib import Path
import shutil
import torch
from typing import Any, Callable, Dict, Optional, Tuple, Union

from osmp.core import VelocityPredictor
from osmp.datasets.auxiliary import ConcatenatedDataset
from osmp.datasets.orbital_datasets import (
    CorneliaTurtleRobotJointSpaceTrajectory,
    Ellipse,
    GreenSeaTurtleSwimmingKinematics,
    ImageContour,
)
from osmp.models.encoder_models import BijectiveEncoder
from osmp.models.dynamics_models import (
    HopfBifurcationDynamics,
    HopfBifurcationWithPhaseSynchronizationDynamics,
)


def orbitally_stable_motion_primitive_factory(
    oracle_name: str,
    model_path: Optional[PathLike] = None,
    num_systems: int = 1,
    device: str = "cpu",
    model_compilation_mode: Optional[str] = "aot.compile",
    saved_model_compilation_mode: Optional[str] = "aot.compile",
    sf: Union[float, np.ndarray] = 1.0,
    sw: Union[float, np.ndarray] = 1.0,
    limit_cycle_kp: float = 1.0,
    phase_sync_kp: float = 0.0,
) -> Tuple[Callable, Any]:
    """
    Factory function to create motion primitives for orbitally stable systems
    Arguments:
        oracle_name: name of the oracle
        model_path: path to the model
        num_systems: number of systems
        device: device to run the model
        model_compilation_mode: mode for compiling the model.
            Options: None, torch.export, torch.compile, aot.compile
        saved_model_compilation_mode: the compilation state of the model to load.
            Options: None, torch.export, aot.compile
        sf: spatial scaling factor
        sw: temporal scaling factor
        limit_cycle_kp: proportional control gain to converge onto the limit cycle
        phase_sync_kp: proportional control gain for phase synchronization
    """
    condition_encoding = False
    encoder_num_blocks = 10  # number of coupling layers
    encoder_num_hidden = 200  # number of random fourier features per block
    scale_vel_num_layers = 3  # number of hidden layers for scaling network
    match oracle_name:
        case "Ellipse":
            dataset = Ellipse(
                a=1.2, b=0.6, max_polar_angle=1.0 * 2 * np.pi, normalize=False
            )
        case "Square" | "Star" | "TUD-Flame" | "MIT-CSAIL" | "Manta-Ray" | "Bat":
            dataset = ImageContour(data_name=oracle_name.lower(), verbose=False)
            encoder_num_blocks = 15
        case "GreenSeaTurtleSwimming" | "GreenSeaTurtleSwimmingWithTwist":
            dataset = GreenSeaTurtleSwimmingKinematics(normalize=True, include_twist_trajectory=True if oracle_name == "GreenSeaTurtleSwimmingWithTwist" else False)
            encoder_num_blocks = 8
            scale_vel_num_layers = 5
        case "CorneliaTurtleRobotJointSpace":
            dataset = CorneliaTurtleRobotJointSpaceTrajectory(normalize=True)
            encoder_num_blocks = 3
            scale_vel_num_layers = 5
        case "CorneliaTurtleRobotJointSpacePitchedDown":
            dataset_nominal = CorneliaTurtleRobotJointSpaceTrajectory(
                normalize=True, smooth=True, verbose=False
            )
            dataset_pitched_down = CorneliaTurtleRobotJointSpaceTrajectory(
                normalize=True, smooth=True, verbose=False
            )
            dataset_pitched_down.x = dataset_pitched_down.x + np.array([0.0, 0.0, -20 / 180 * np.pi], dtype=np.float32)
            dataset = ConcatenatedDataset([dataset_nominal, dataset_pitched_down])
            dataset.scaling = dataset_nominal.scaling
            dataset.translation = dataset_nominal.translation
            condition_encoding = True
            encoder_num_blocks = 3
            scale_vel_num_layers = 5
        case _:
            raise ValueError(f"Unknown oracle name: {oracle_name}")
    num_dims = dataset.n_dims
    sample_input = torch.ones(
        (num_systems, num_dims), requires_grad=False, device=device
    )
    sample_args = (sample_input,)
    sample_kwargs = dict()
    if condition_encoding:
        sample_kwargs["z"] = torch.ones(
            (num_systems, 1), requires_grad=False, device=device
        )
    sample_dynamics_kwargs = dict(
        alpha=sw * torch.tensor(limit_cycle_kp, device=device).float(),
        omega=torch.tensor(sw, device=device).float(),
        beta=sw * torch.tensor(limit_cycle_kp, device=device).float(),
    )
    if phase_sync_kp > 0.0:
        sample_dynamics_kwargs["phase_sync_kp"] = torch.tensor(
            phase_sync_kp, device=device
        ).float()
    if len(sample_dynamics_kwargs) > 0:
        sample_kwargs["dynamics_kwargs"] = sample_dynamics_kwargs

    if model_path is None:
        import osmp

        model_dir = Path(osmp.__file__).parent.parent.parent / "models"
        model_name_postfix = "_phase_sync" if phase_sync_kp > 0.0 else ""
        model_name = f"{oracle_name}_hopf_bifurcation_rffn{model_name_postfix}"
        match saved_model_compilation_mode:
            case None:
                model_path = (model_dir / (model_name + ".pt")).resolve()
            case "torch.export" | "torch.compile":
                model_path = (model_dir / (model_name + ".pt2")).resolve()
            case "aot.compile":
                model_path = (model_dir / (model_name + f".so")).resolve()
            case _:
                raise ValueError

    model = None
    if model_path.suffix == ".pt":
        taskmap_net = BijectiveEncoder(
            num_dims=num_dims,
            num_blocks=encoder_num_blocks,
            num_hidden=encoder_num_hidden,
            jacobian_computation_method="numerical",
            condition=condition_encoding,
        )

        if phase_sync_kp > 0.0:
            dynamics_model = HopfBifurcationWithPhaseSynchronizationDynamics(
                state_dim=num_dims
            )
        else:
            dynamics_model = HopfBifurcationDynamics(state_dim=num_dims)

        # pulled back dynamics (natural gradient descent system)
        model = VelocityPredictor(
            taskmap_model=taskmap_net,
            dynamics_model=dynamics_model,
            origin=None,
            scale_vel=True,
            scale_vel_num_layers=scale_vel_num_layers,
            is_diffeomorphism=True,
            n_dim_x=num_dims,
            n_dim_y=num_dims,
            x_translation=torch.from_numpy(dataset.translation).float(),
            x_scaling=torch.from_numpy(dataset.scaling).float(),
        )

        # Loading learner
        print(f"Loading the bare model from {model_path} ...")
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )
        model.eval()

        if model_compilation_mode is not None:
            # export the model
            # https://pytorch.org/tutorials/intermediate/torch.export_tutorial.html
            print(f"Executing torch.export ...")
            ep = torch.export.export(
                model, args=sample_args, kwargs=sample_kwargs, strict=True
            )

            # save the exported model
            saved_ep_path = (
                model_path.parent / f"{model_path.stem}_ns_{num_systems}.pt2"
            )
            torch.export.save(ep, saved_ep_path)
            print(f"Exported model saved to: {saved_ep_path}")

            model = ep.module()

    elif model_path.suffix == ".pt2":
        print(f"Loading the exported model from {model_path} ...")
        ep = torch.export.load(str(model_path))
        model = ep.module()

    if model_compilation_mode == "torch.compile":
        print(f"Executing torch.compile ...")
        with torch.no_grad():
            model = torch.compile(model, fullgraph=True)
            for _ in range(10):
                _ = model(*sample_args, **sample_kwargs)
            print("The model warmed up now!")
    elif model_path.suffix == ".so":
        print(f"Loading the aot compiled model from {model_path} ...")
        model = torch._export.aot_load(str(model_path), device=device)
    elif model_compilation_mode == "aot.compile":
        # Compile the exported program to a .so using ``AOTInductor``
        print("Executing aot.compile ...")
        with torch.no_grad():
            so_path = torch._inductor.aot_compile(
                ep.module(), args=sample_args, kwargs=sample_kwargs
            )

        # copy the compiled model to the models directory
        shutil.copy(so_path, model_path.with_suffix(".so"))
        print(f"Compiled AOT model saved to: {model_path.with_suffix(".so")}")

        # Load and run the .so file in Python.
        model = torch._export.aot_load(so_path, device=device)

    def motion_primitive_fn(
        t: Union[float, np.ndarray], x: np.array, z: Optional[np.array] = None
    ) -> Tuple[np.array, Dict[str, np.array]]:
        """
        Control function that takes in the time and the current position and returns the velocity
        Arguments:
            t: the current time
            x: the current positions as shape (num_systems, num_dims) numpy array
            z: the encoding conditioning variable as shape (num_systems, num_dims) numpy array
        Returns:
            x_d: the velocities as a numpy array with shape (num_systems, num_dims)
            aux: dictionary of auxiliary outputs
        """
        # initialize auxiliary outputs
        aux = dict(x_unnorm=x)
        if z is not None:
            aux["z"] = z

        # subtract the dataset translation and scale the position
        # i.e., normalize the position
        x = x * dataset.scaling / sf + dataset.translation
        aux["x_norm"] = x

        with torch.no_grad():
            # convert to torch tensor
            x = torch.tensor(x, requires_grad=False).float()

            kwargs = dict()
            if z is not None:
                kwargs["z"] = torch.tensor(z, device=device).float()
            dynamics_kwargs = dict(
                alpha=sw * torch.tensor(limit_cycle_kp, device=device).float(),
                omega=torch.tensor(sw, device=device).float(),
                beta=sw * torch.tensor(limit_cycle_kp, device=device).float(),
            )
            if phase_sync_kp > 0.0:
                dynamics_kwargs["phase_sync_kp"] = torch.tensor(
                    phase_sync_kp, device=device
                ).float()
            if len(dynamics_kwargs) > 0:
                kwargs["dynamics_kwargs"] = dynamics_kwargs

            # evaluate the trained model
            x_d, aux_model = model(x, **kwargs)
            for key in aux_model:
                aux_model[key] = aux_model[key].cpu().numpy()
            aux = aux | aux_model

        # transfer to numpy
        x_d = x_d.cpu().numpy()
        aux["x_d_norm"] = x_d

        # denormalize the velocity
        x_d = x_d / dataset.scaling * sf

        # update the auxiliary outputs
        aux["x_d_unnorm"] = x_d

        return x_d, aux

    return motion_primitive_fn, model
