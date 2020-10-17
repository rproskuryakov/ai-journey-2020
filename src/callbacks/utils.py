from typing import Callable


def init_is_better(mode: str, min_delta: float, percentage: bool) -> Callable:
    if mode not in {'min', 'max'}:
        raise ValueError(f'mode {mode} is unknown!')
    if not percentage:
        if mode == 'min':
            is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            is_better = lambda a, best: a > best + min_delta
    else:
        if mode == 'min':
            is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
        if mode == 'max':
            is_better = lambda a, best: a > best + (
                    best * min_delta / 100)
    return is_better
