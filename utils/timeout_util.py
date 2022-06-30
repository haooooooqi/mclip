from absl import logging
from utils import logging_util
import signal
from jax.experimental import multihost_utils
import jax
import time
import os
import sys


def timeout(signum, frame):
  logging_util.verbose_on()
  logging.info('Time out.')
  # logging_util.verbose_off()
  # try:
  #   multihost_utils.sync_global_devices(f'timeout')
  # except:
  #   sys.exit()
  sys.exit()


def timeout_on(seconds):
  signal.signal(signal.SIGALRM, timeout)
  signal.alarm(seconds)


def timeout_off():
  signal.alarm(0)

