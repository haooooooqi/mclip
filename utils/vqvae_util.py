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

  def __call__(self, inputs, train=True, split=False):
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

    if train:
      counts = jnp.zeros(self.vocab_size, dtype=jnp.int32)
      counts = counts.at[ids].add(1)

      from IPython import embed; embed();
      if (0 == 0): raise NotImplementedError
      x_sum = jnp.zeros_like(self.dictionary.value)
      x_sum = x_sum.at[ids].add(jax.lax.stop_gradient(x_pre_q))


    # ------------------------------------------------
    # compute the "codes"
    encoding_indices = jnp.argmax(-distances, axis=-1)  # nearest-neighbor encoding, (M,)
    encodings = jax.nn.one_hot(encoding_indices, self.vocab_size)  # one-hot, (M, K)

    # compute "quantized" x
    quantized = jnp.einsum('mk,ck->mc', encodings, emb)  # (M, C), same as x_flat
    quantized = quantized.reshape(x.shape)  # (.., .., C)

    # compute the VQ loss
    # reduce_mean: this is l2_dist / D
    e_latent_loss = ((jax.lax.stop_gradient(quantized) - x)**2).mean()
    q_latent_loss = ((quantized - jax.lax.stop_gradient(x))**2).mean()
    loss_vq = q_latent_loss + self.beta * e_latent_loss

    # straight-through estimator
    quantized = x + jax.lax.stop_gradient(quantized - x)

    # compute the perplexity for monitoring
    avg_probs = encodings.mean(axis=0)  # usage of each embedding, (K,)
    avg_probs = dist_util.pmean(avg_probs, axis_name='batch')

    if train:
      running_avg_probs = self.running_avg_probs
      running_avg_probs.value = running_avg_probs.value * self.momentum + avg_probs * (1 - self.momentum)
      avg_probs = running_avg_probs.value

    perplexity = jax.lax.exp(-jnp.sum(avg_probs * jax.lax.log(avg_probs + 1e-10)))

    return quantized, loss_vq, perplexity


def split_embeddings(emb, probs, cfg):
  """
  emb: [D, K]
  probs: [K,] 
  """
  ids_max = jnp.argmax(probs)
  ids_min = jnp.argmin(probs)

  prob_max = probs[ids_max]
  prob_min = probs[ids_min]
  
  thr = cfg.threshold
  if prob_max > prob_min * thr:
    logging.info('Splitting...')

    emb_max = emb[:, ids_max]
    emb_min = emb[:, ids_min]
    emb_max_jit = emb_max + 1e-6 * emb_min  # hack: slightly change it towards emb_min

    new_emb = emb.at[:, ids_min].set(emb_max_jit)
    return new_emb
  else:
    return emb
