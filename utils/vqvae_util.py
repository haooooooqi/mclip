import jax
import flax
import jax.numpy as jnp

import flax.linen as nn


class VectorQuantizer(nn.Module):
  vocab_size: int
  beta: float

  @nn.compact
  def __call__(self, x):
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
    perplexity = jax.lax.exp(-jnp.sum(avg_probs * jax.lax.log(avg_probs + 1e-10)))

    return quantized, loss_vq, perplexity