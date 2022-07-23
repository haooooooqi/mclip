from ast import Not
import jax
import flax
import jax.numpy as jnp


def GumbelSoftmaxLoss(logits, rng, kl_weight=1.0, temperature=1.0, is_hard=False):
  """
  logits: [.., .., K]
  """
  input_shape = logits.shape
  K = input_shape[-1]

  logits = logits.reshape([-1, K])  # [M, K]

  g = gumbel_softmax(logits, rng, temperature, is_hard)  # gumbel_softmax scores, (M, K)


def gumbel_softmax(logits, rng, temperature, is_hard):
  gumbel_softmax_sample = logits + jax.random.gumbel(rng, logits.shape)
  y = jax.nn.softmax(gumbel_softmax_sample / temperature, axis=-1)

  if is_hard:
    raise NotImplementedError
  
  return y