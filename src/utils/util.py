import logging
import pathlib
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from tqdm import tqdm

from pytorch_lightning.utilities import rank_zero_only


class Object(object):
    pass


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


HORSE = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;;;;;'. ;   |          )   \\ | ;;;;;;
          ,;*;;;;\\/   |.        /   /` | ';;;*;
           ;OptML/    |/       /   /__/   ';;;
           '*;;;/     |       /    |      ;*;
                `""""`        `""""`     ;'"""


def nice_print(msg, last=False):
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


def ensure_dir(dirname: str):
    """
    Ensure that the given path is a directory and create it if it does not exist.

    :param dirname: The path to the directory.
    """
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# heavily inspired by
# https://moonbooks.org/Articles/How-to-numerically-compute-the-inverse-function-in-python-using-scipy-/
def get_numerical_approx_inverse_focal_loss(gamma, start=0.01, resolution=0.01):
    def focal_loss(prob):
        y = -(1 - prob) ** gamma * np.log(prob)
        return y

    def diff(x, a):
        yt = focal_loss(x)
        return (yt - a) ** 2

    f_preimage = np.arange(start, 1.0, resolution)
    f_image = focal_loss(f_preimage)

    g_preimage = np.arange(np.min(f_image), np.max(f_image), resolution)
    g_image = np.zeros(g_preimage.shape)

    for idx, x_value in enumerate(tqdm(g_preimage)):
        res = minimize(diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
        g_image[idx] = res.x[0]

    return f_image, g_image


def approx_probs(loss, gamma, f_start=0.01, resolution=0.001):
    f_image, g_image = get_numerical_approx_inverse_focal_loss(gamma, start=f_start, resolution=resolution)
    g_indices = ((loss - f_image.min()) / resolution).astype(int)
    g_indices = np.minimum(g_indices, g_image.shape[0] - 1)
    return g_image[g_indices]


log = get_logger(__name__)
