import jax
import flax
import jax.numpy as jnp

from absl import logging

import flax.linen as nn
from utils import dist_util


class VectorQuantizer(nn.Module):
  vocab_size: int
  beta: float

  @nn.compact
  def __call__(self, x, train=True):
    """
    Input:
    x: [.., .., C]
    Output:
    q: [.., .., C] of the same shape
    """

    input_shape = x.shape
    C = input_shape[-1]

    emb = self.param('vq_embed', nn.initializers.xavier_uniform(), [C, self.vocab_size])

    x_flat = x.reshape([-1, C])  # (M, C)

    distances = (
      (x_flat**2).sum(axis=1, keepdims=True)  # (M, 1)
      - 2 * jnp.einsum('mc,ck->mk', x_flat, emb)  # (M, K)
      + (emb**2).sum(axis=0, keepdims=True)  # (1, K)
    )  # (M, K)

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

    # the ema update version
    running_avg_probs = self.variable('vqvae', 'running_avg_probs', lambda s: jnp.ones(s, jnp.float32) / s[0], (self.vocab_size,))
    if train:
      momentum = 0.9
      running_avg_probs.value = running_avg_probs.value * momentum + avg_probs * (1 - momentum)
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
