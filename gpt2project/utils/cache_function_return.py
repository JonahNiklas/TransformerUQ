import logging
import os
import functools
from typing import Any, Callable, TypeVar

import torch

from gpt2project.uq.evaluation_run_config import EvaluationRunConfig
from utils.general_plotter import get_gpt_evaluation_path

T = TypeVar("T")

logger = logging.getLogger(__name__)


def cache_evaluation_run_return() -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that caches the return value of a function to a path determined by the evaluation_run_config.
    The function must take an EvaluationRunConfig as an argument.

    Returns:
        Decorated function that caches its return value
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Extract evaluation_run_config from args or kwargs
            evaluation_run_config = None
            for arg in args:
                if isinstance(arg, EvaluationRunConfig):
                    evaluation_run_config = arg
                    break

            if evaluation_run_config is None:
                raise ValueError(
                    "Function must be called with an EvaluationRunConfig parameter"
                )

            # Get cache path from the evaluation_run_config
            cache_path = get_gpt_evaluation_path(evaluation_run_config)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            assert cache_path.endswith(".pt"), "Cache path must end with .pt"
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        result = torch.load(f)
                    logger.info(f"Loaded result from cache: {cache_path}")
                    return result  # type: ignore
                except Exception as e:
                    logger.warning(f"Error loading cache: {e}. Computing result...")

            # Compute the result if cache doesn't exist or couldn't be loaded
            result = func(*args, **kwargs)

            # Save result to cache
            try:
                with open(cache_path, "wb") as f:
                    torch.save(result, f)
                logger.info(f"Saved result to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Error saving to cache: {e}")

            return result  # type: ignore

        return wrapper

    return decorator
