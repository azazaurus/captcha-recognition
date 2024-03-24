import onelayermnist as mnist
from argparse import ArgumentParser


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def initialize_argument():
    parser = ArgumentParser(
        "MNIST digit recognition with convolutional SNN. Requires Tensorboard, Matplotlib, and Torchvision"
    )
    parser.add_argument(
        "--only-first-spike",
        type=bool,
        default=False,
        help="Only one spike per input (latency coding).",
    )
    parser.add_argument(
        "--save-grads",
        type=bool,
        default=False,
        help="Save gradients of backward pass.",
    )
    parser.add_argument(
        "--grad-save-interval",
        type=int,
        default=10,
        help="Interval for gradient saving of backward pass.",
    )
    parser.add_argument(
        "--refrac", type=bool, default=False, help="Use refractory time."
    )
    parser.add_argument(
        "--plot-interval", type=int, default=10, help="Interval for plotting."
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Scaling factor for input current.",
    )
    parser.add_argument(
        "--find-learning-rate",
        type=bool,
        default=False,
        help="Use learning rate finder to find learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use by pytorch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training episodes to do."
    )
    parser.add_argument(
        "--seq-length", type=int, default=200, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of examples in one minibatch.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="super",
        choices=["super", "tanh", "circ", "logistic", "circ_dist"],
        help="Method to use for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to use for saving the results"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--clip-grad",
        type=bool,
        default=False,
        help="Clip gradient during backpropagation",
    )
    parser.add_argument(
        "--grad-clip-value", type=float, default=1.0, help="Gradient to clip at."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="Learning rate to use."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="In which intervals to display learning progress.",
    )
    parser.add_argument(
        "--model-save-interval",
        type=int,
        default=50,
        help="Save model every so many epochs.",
    )
    parser.add_argument(
        "--save-model", type=bool, default=True, help="Save the model after training."
    )
    parser.add_argument("--big-net", type=bool, default=False, help="Use bigger net...")
    parser.add_argument(
        "--only-output", type=bool, default=False, help="Train only the last layer..."
    )
    parser.add_argument(
        "--do-plot", type=bool, default=False, help="Do intermediate plots"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1234, help="Random seed to use"
    )
    args = parser.parse_args()
    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mnist.main(initialize_argument())

