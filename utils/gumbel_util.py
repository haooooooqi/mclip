from ast import Not
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from typing import Any, Callable, Optional, Tuple


class GumbelVectorQuantizer(nn.Module):
  gumbel: Any  # config of gumbel

  @nn.compact
  def __call__(self, x, rng):
    """
    Input:
    x: [.., .., C]
    Output:
    q: [.., .., C] of the same shape
    """
    # prepare the input
    input_shape = x.shape
    C = input_shape[-1]
    x_flat = x.reshape([-1, C])  # (M, C)

    # project x
    emb = self.param('vq_embed', nn.initializers.xavier_uniform(), [C, self.gumbel.vocab_size])

    logits = jnp.einsum('mc,ck->mk', x_flat, emb)  # (M, K), bigger is more similar

    softmax_logits, kl_div, perplexity = GumbelSoftmaxWithLoss(logits, rng, tau=self.gumbel.tau, is_hard=self.gumbel.is_hard)

    if self.gumbel.softmax_only:
      softmax_logits = jax.nn.softmax(logits / self.gumbel.tau, axis=-1)

    quantized = jnp.einsum('mk,ck->mc', softmax_logits, emb)  # (M, C), same as x_flat
    quantized = quantized.reshape(x.shape)  # (.., .., C)

    return quantized, kl_div, perplexity


def GumbelSoftmaxWithLoss(logits, rng, tau=1.0, is_hard=False):
  """
  logits: [.., .., K]
  """
  input_shape = logits.shape
  K = input_shape[-1]

  logits = logits.reshape([-1, K])  # [M, K]

  g = gumbel_softmax(logits, rng, tau, is_hard)  # gumbel_softmax scores, (M, K)
  g = g.reshape(input_shape)  # [N, ..., K]

  # compute the KL div loss for regularization
  # v2:
  q = jax.nn.softmax(logits)  # (N*L, K)
  avg_q = q.mean(axis=0)  # (K,)
  kl_div = avg_q * (jax.lax.log(avg_q) - jax.lax.log(1.0 / K))  # (K,)
  kl_div = kl_div.sum()

  # compute the perplexity
  encoding_indices = jnp.argmax(logits, axis=-1)  # nearest-neighbor encoding, (M,)
  encodings = jax.nn.one_hot(encoding_indices, K)  # one-hot, (M, K)
  avg_probs = encodings.mean(axis=0)  # usage of each embedding, (K,)
  perplexity = jax.lax.exp(-jnp.sum(avg_probs * jax.lax.log(avg_probs + 1e-10)))

  return g, kl_div, perplexity


def gumbel_softmax(logits, rng, tau, is_hard):
  gumbels = logits + jax.random.gumbel(rng, logits.shape)
  y = jax.nn.softmax(gumbels / tau, axis=-1)

  if is_hard:
    z = jnp.max(y, axis=-1, keepdims=True)
    y_hard = jnp.float32(y == z)
    y = jax.lax.stop_gradient(y_hard - y) + y
  
  return y