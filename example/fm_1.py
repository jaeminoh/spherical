import jax
import equinox as eqx
from jaxtyping import Array, Float
import optax, jaxopt
import jax.numpy as jnp
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np


class Flow(eqx.Module):
    layers: list

    def __init__(self, dim: int = 2, h: int = 64):
        self.layers = [eqx.nn.Linear()]