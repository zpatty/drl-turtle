__all__ = ["OsmpArgumentParser"]
import argparse


class OsmpArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(OsmpArgumentParser, self).__init__(
            description="Orbitally Stable Motion Primitives", **kwargs
        )

        self.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "cuda"],
            help="Device to run the model on.",
        )

        self.add_argument(
            "--oracle-name",
            type=str,
            default="CorneliaTurtleRobotJointSpace",
            help="Name of the oracle/dataset.",
        )

        self.add_argument(
            "--dynamics-type",
            type=str,
            default="hopf_bifurcation",
            choices=[
                "linear",
                "van_der_pol_oscillator",
                "hopf_bifurcation",
                "mlp",
                "rnn",
                "gru",
                "lstm",
                "cornn",
                "lem",
            ],
            help="Type of latent space dynamics.",
        )

        self.add_argument(
            "--encoder-type",
            type=str,
            default="rffn",
            choices=[
                "id",
                "rffn",
                "fcnn",
                "normflows_real_nvp",
                "normflows_neural_spline_flow",
            ],
            help="Type of encoder network.",
        )
