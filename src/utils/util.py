import logging
import pathlib
from datetime import datetime

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


log = get_logger(__name__)
