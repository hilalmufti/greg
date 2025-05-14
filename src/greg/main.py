# %%
import time
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
import mlx.core as mx

# %%
LOGO = [
    "   ____ _________  ____ _",
    "  / __ `/ ___/ _ \\/ __ `/",
    " / /_/ / /  /  __/ /_/ / ",
    " \\__, /_/   \\___/\\__, /  ",
    "/____/          /____/   "
]

# %%
def show_list_by(delim: str, xs: list):
    assert isinstance(xs, list)

    n = ndims(xs)
    if n == 1:
        return delim.join(map(str, xs))
    elif n == 2:
        return "\n".join(map(lambda x: show_list_by(delim, x), xs))
    else:
        raise Exception()

# %%
show_list = partial(show_list_by, " ")

# %%
def isnumeric(s):
    if s.startswith("-"):
        return isnumeric(s[1:])
    else:
        return s.isnumeric() or (len((ts:=s.split("."))) <= 2 and all(t.isnumeric() for t in ts))

# %%
def show_float(x):
    return f"{x:.15f}"

# %%
def read_list(s):
    return [[float(x) if isnumeric(x) else x for x in l.split()] for l in s.split("\n")]

# %%
def mse_loss(weights, fwd, inputs, targets):
    return 0.5 * mx.mean(mx.square(fwd(inputs, weights) - targets))

# %%
def sgd_solver(lr):
    def sgd_step(weights, grads):
        return weights - lr * grads
    return sgd_step

# %%
def linreg_weights(n_features, key):
    return 1e-2 * mx.random.normal([n_features], key=key)

# %%
def linreg_forward(inputs, weights):
    return mx.matmul(inputs, weights)

# %%
def solve(weights, forward, loss_fn, solver_step, inputs, targets, n_iters):
    grad_fn = mx.grad(loss_fn)
    for _ in range(n_iters):
        grads = grad_fn(weights, forward, inputs, targets)
        weights = solver_step(weights, grads)
        mx.eval(weights)
    return weights

# %%
def ndims(xs):
    match xs:
        case []:
            return 1
        case [x, *xs] if not isinstance(x, list):
            return 1
        case [x, *xs] if isinstance(x, list):
            return 1 + ndims(x)

# %%
def unzip_data(dataset):
    return dataset[:,:-1], dataset[:,-1]


# %%
def run_experiment(config):
    train_inputs = config["train_inputs"]
    train_targets = config["train_targets"]
    test_inputs = config["test_inputs"]
    test_targets = config["test_targets"]

    seed = config["seed"]
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    n_iters = config["n_iters"]
    lr = config["lr"]
    make_weights = config["make_weights"]
    forward = config["forward"]
    loss_fn = config["loss_fn"]
    make_solver = config["make_solver"]
    solver_step = make_solver(lr)


    key = mx.random.key(config["seed"])

    key, subkey = mx.random.split(key)
    weights = make_weights(n_features, subkey)

    tic = time.perf_counter()
    weights = solve(weights, forward, loss_fn, solver_step, train_inputs, train_targets, n_iters)
    toc = time.perf_counter()

    train_loss = loss_fn(weights, forward, train_inputs, train_targets).item()
    throughput = (toc - tic) / n_iters
    if test_inputs is not None and test_targets is not None:
        test_loss = loss_fn(weights, forward, test_inputs, test_targets).item()
    else:
        test_loss = None
    return {
        "weights": weights,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "throughput": throughput
    }

# %%
def main() -> None:
    parser = ArgumentParser(prog="greg",
                            description="greg, the gENERAL regRESSION program, applies linear regression to your dataset",
                            epilog="\n".join(LOGO),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--seed",
                        type=int,
                        default=546,
                        help="Specify the random seed used to initialize initial random weights")
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10_000,
                        help="Specify the number of training iterations through the entire dataset")
    parser.add_argument("-a", "--all",
                        action="store_true",
                        help="Show all metrics after training")
    args = parser.parse_args()

    slurped = sys.stdin.read().strip()

    samples = read_list(slurped)
    if samples[0][0] in ["train", "test"]:
        train_samples = [sample[1:] for sample in samples if sample[0] == "train"]
        test_samples = [sample[1:] for sample in samples if sample[0] == "test"]

        train_inputs, train_targets = unzip_data(mx.array(train_samples))
        test_inputs, test_targets = unzip_data(mx.array(test_samples))
    else:
        train_inputs, train_targets = unzip_data(mx.array(samples))
        test_inputs, test_targets = None, None

    config = {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "test_inputs": test_inputs,
        "test_targets": test_targets,
        "seed": args.seed,
        "n_samples": train_inputs.shape[0],
        "n_features": train_inputs.shape[-1],
        "n_iters": args.epochs,
        "lr": 1e-2,
        "forward": linreg_forward,
        "make_weights": linreg_weights,
        "loss_fn": mse_loss,
        "make_solver": sgd_solver
    }

    results = run_experiment(config)
    weights = results["weights"]
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    throughput = results["throughput"]

    if args.all:
        print("train/loss", show_float(train_loss))
        print("train/throughput", show_float(throughput))
        if test_loss is not None:
            print("test/loss", show_float(test_loss))
    print(show_list(weights.tolist()))

# TODO:
# - [ ] feat: add int support to read_list()
# - [ ] feat: add support to parsing scientific notation
# - [ ] feat: add support for multiple output dimensions
# - [ ] feat: stop copy-pasting code from `xyn`
# - [x] feat: add support for train/test sets
# - [x] feat: add experiment launcher
# - [ ] feat: add support for reading data from a file
# - [ ] feat: optimize solve() via jit compilation
# - [ ] feat: add logging for entire solving process

# Rambling
# If your problem space is continuous:
# - optimizer
# - train
# If your problem space is discrete:
# - solver
# - solve
