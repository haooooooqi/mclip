from absl import logging
from utils import logging_util
import signal
from jax.experimental import multihost_utils
import time


def timeout(signum, frame):
  logging_util.verbose_on()
  logging.info('Time out.')
  logging_util.verbose_off()
  # multihost_utils.sync_global_devices(f'timeout')
  time.sleep(10)
  exit()
  # raise RuntimeError


def timeout_on(seconds):
  signal.signal(signal.SIGALRM, timeout)
  signal.alarm(seconds)


def timeout_off():
  signal.alarm(0)

