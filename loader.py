from typing import Generator, NamedTuple, Optional
from jaxtyping import Array, Float, Int

import jax.numpy as jnp
import jax.random as jr
import numpy as np


class Batch(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


def num_batches(x: jnp.ndarray, batch_size: int) -> int:
    """
    Compute the number of batches in an array.

    Args:
        x: The array to compute the number of batches in.
        batch_size: The size of each batch.

    Returns:
        The number of batches in the array.
    """
    num_complete_batches, leftover = divmod(x.shape[0], batch_size)
    return num_complete_batches + bool(leftover)


def num_batches(arr: Array, batch_size: int, axis: int = 0) -> int:
    num_full, leftover = divmod(arr.shape[axis], batch_size)
    return num_full + bool(leftover)


def data_loader(
    x: Float[Array, "n x1 x2 ch"],
    y: Int[Array, "n"],
    batch_size: int,
    shuffle: Optional[jr.KeyArray] = None,
):
    perm = np.arange(x.shape[0])
    if shuffle is not None:
        np.random.shuffle(perm)

    for i in range(num_batches(x, batch_size)):
        batch_idx = perm[i * batch_size : (i + 1) * batch_size]
        yield Batch(x[batch_idx], y[batch_idx])


class DataLoader:
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        batch_size: int,
        shuffle: Optional[jr.KeyArray] = None,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Generator[Batch, None, None]:
        return data_loader(self.x, self.y, self.batch_size, self.shuffle)

    def __len__(self) -> int:
        return num_batches(self.x, self.batch_size)

    def __getitem__(self, idx: int) -> Batch:
        if idx >= len(self):
            raise IndexError
        return Batch(self.x[idx], self.y[idx])
