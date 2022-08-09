import jax
import flax
import jax.numpy as jnp

from absl import logging

import flax.linen as nn
from utils import dist_util


class VectorQuantizer(nn.Module):
  vocab_size: int
  beta: float
  dim: int
  momentum: float = 0.995
  # Threshold for the discounted count after which the codeword will be
  # considered unused. For the `dict_momentum` param of 0.995 the codeword
  # should not be present in ~500 batches in a row.
  min_count: float = 0.1  # ~= 0.995 ** 500

  def setup(self):
    # the ema update version
    norm_init = nn.initializers.normal(stddev=1.0 / (self.vocab_size**.5))
    self.dictionary = self.variable('vqvae', 'dictionary',
      lambda shape: norm_init(self.make_rng("params"), shape),
      (self.vocab_size, self.dim,))
    self.counts = self.variable('vqvae', 'counts', lambda shape: jnp.ones(shape, jnp.float32), (self.vocab_size,))

  def get_codewords(self):
    e = self.dictionary.value / self.counts.value[:, None]
    return e

  @staticmethod
  def quantize(x, e):
    """
    x: [M, C]
    e: [K, C]
    """
    distances = (
      (x**2).sum(axis=1, keepdims=True)  # (M, 1)
      - 2 * jnp.einsum('mc,kc->mk', x, e)  # (M, K)
      + (e**2).sum(axis=1)  # (K,)
    )  # (M, K)
    ids = jnp.argmin(distances, axis=-1)  # nearest-neighbor encoding, (M,)

    one_hot = jax.nn.one_hot(ids, e.shape[0])  # one-hot, (M, K)
    x_q = jnp.einsum('mk,kc->mc', one_hot, e)  # (M, C), same as x

    x_q = x + jax.lax.stop_gradient(x_q - x)  # straight-through estimator
    return x_q, ids

  def __call__(self, inputs, train=True):
    """
    Input:
    inputs: [.., .., C]
    Output:
    q: [.., .., C] of the same shape
    """

    input_shape = inputs.shape
    C = input_shape[-1]

    x = inputs.reshape([-1, C])  # (M, C)

    e = self.get_codewords()
    x_pre_q = x
    x_q, ids = self.quantize(x, e)

    # perform k-means like update
    if train:
      counts = jnp.zeros(self.vocab_size, dtype=jnp.int32)
      counts = counts.at[ids].add(1)

      x_sum = jnp.zeros_like(self.dictionary.value)
      x_sum = x_sum.at[ids].add(jax.lax.stop_gradient(x_pre_q))

      counts = dist_util.psum(counts, axis_name='batch')
      x_sum = dist_util.psum(x_sum, axis_name='batch')

      # update dict
      self.counts.value = self.counts.value * self.momentum + counts
      self.dictionary.value = self.dictionary.value * self.momentum + x_sum

      state = {"dictionary": self.dictionary.value,
                "counts": self.counts.value,
                "rng": self.make_rng("vqvae"),  # this should be a device-shared rng
                "step": jnp.zeros((), dtype=jnp.int32)}
      # split_the_most_frequent_embedding(state)
      new_state = jax.lax.while_loop(
          lambda state: jnp.logical_and(jnp.any(state["counts"] < self.min_count), state["step"] < 100),
          split_the_most_frequent_embedding,
          state)
      self.counts.value = new_state["counts"]
      self.dictionary.value = new_state["dictionary"]

    # compute the VQ loss
    # reduce_mean: this is l2_dist / D
    e_latent_loss = ((jax.lax.stop_gradient(x_q) - x)**2).mean()
    # q_latent_loss = ((x_q - jax.lax.stop_gradient(x))**2).mean()
    loss_vq = self.beta * e_latent_loss

    # compute the perplexity for monitoring
    avg_probs = self.counts.value
    avg_probs /= avg_probs.sum()
    perplexity = jax.lax.exp(-jnp.sum(avg_probs * jax.lax.log(avg_probs + 1e-10)))

    x_q = x_q.reshape(input_shape)
    return x_q, loss_vq, perplexity


def split_the_most_frequent_embedding(state):
  """Splits most frequent embedding into two and eliminates least frequent.

  Args:
    state: a dict. that contains current jax rng, embeddings and their counts.

  Returns:
    New dict. with the updated jax rng, embeddings and counts.
  """
  rng, e, c, step = state["rng"], state["dictionary"], state["counts"], state["step"]
  rng, rng_local = jax.random.split(rng)

  i_max = jnp.argmax(c)
  i_min = jnp.argmin(c)

  PERTURB = 0.001
  jitter = jax.random.uniform(rng_local, (e.shape[1],), jnp.float32, 1.0 - PERTURB, 1.0 + PERTURB)
  e = e.at[i_min].set(e[i_max] * jitter)

  c = c.at[i_min].set(c[i_max] / 2.0)
  c = c.at[i_max].set(c[i_max] / 2.0)

  # dictionary is the sum of centers, so when c is changed, e should be changed
  e = e.at[i_min].set(e[i_min] / 2.0)
  e = e.at[i_max].set(e[i_max] / 2.0)

  step += 1
  return {"rng": rng, "dictionary": e, "counts": c, "step": step}