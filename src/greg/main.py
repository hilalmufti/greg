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
def show_float(x):
    return f"{x:.15f}"

# %%
def read_list(s):
    return [[float(x) for x in l.split()] for l in s.split("\n")]

# %%
def mse_loss(weights, fwd, inputs, targets):
    return 0.5 * mx.mean(mx.square(fwd(inputs, weights) - targets))

# %%
def sgd_solver(lr):
    def sgd_step(weights, grads):
        return weights - lr * grads
    return sgd_step

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

    dataset = mx.array(read_list(slurped))
    inputs = dataset[:,:-1]
    targets = dataset[:,-1]

    seed = args.seed
    n_samples = inputs.shape[0]; assert n_samples == targets.shape[0]
    n_features = inputs.shape[-1]
    n_iters = args.epochs
    lr = 1e-2
    forward = linreg_forward
    loss_fn = mse_loss
    solver_step = sgd_solver(lr)

    key = mx.random.key(seed)

    # construct initial random weights
    key, subkey = mx.random.split(key)
    weights = 1e-2 * mx.random.normal([n_features], key=subkey)

    tic = time.perf_counter()
    weights = solve(weights, forward, loss_fn, solver_step, inputs, targets, n_iters) # train the regression model
    toc = time.perf_counter()

    if args.all:
        loss = loss_fn(weights, forward, inputs, targets).item()
        throughput = n_iters / (toc - tic)
        print("loss", show_float(loss))
        print("throughput", show_float(throughput)) # it/s
    print(show_list(weights.tolist()))

# TODO:
# - [ ] feat: add int support to read_list()
# - [ ] feat: add support for multiple output dimensions
# - [ ] feat: stop copy-pasting code from `xyn`
# - [ ] feat: add support for train/test sets
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
