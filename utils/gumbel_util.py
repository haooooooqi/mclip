from ast import Not
import jax
import flax
import jax.numpy as jnp


def GumbelSoftmaxLoss(logits, rng, temperature=1.0, is_hard=False):
  """
  logits: [.., .., K]
  """
  input_shape = logits.shape
  K = input_shape[-1]

  logits = logits.reshape([-1, K])  # [M, K]

  g = gumbel_softmax(logits, rng, temperature, is_hard)  # gumbel_softmax scores, (M, K)

  from IPython import embed; embed();
  if (0 == 0): raise NotImplementedError

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


def gumbel_softmax(logits, rng, temperature, is_hard):
  gumbel_softmax_sample = logits + jax.random.gumbel(rng, logits.shape)
  y = jax.nn.softmax(gumbel_softmax_sample / temperature, axis=-1)

  if is_hard:
    raise NotImplementedError
  
  return y