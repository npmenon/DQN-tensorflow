from __future__ import print_function

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import random
import tensorflow as tf
from keras import backend as K
import threading

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction


def worker(agent, env, num):
  print("********************** Starting thread ", num + 1, " **************************", end="\n")
  agent.train(env, threading.currentThread().ident)


def init_threads(agent, config):
  envs = [GymEnvironment(config) for _ in range(config.number_of_threads)]
  actor_learner_threads = [threading.Thread(target=worker, args=(agent, envs[i], i)) for i in range(config.number_of_threads)]
  for i in range(config.number_of_threads):
    actor_learner_threads[i].start()

  # if config.display:
  #   while True:
  #     for env in envs:
  #       env.render()

  for i in range(config.number_of_threads):
    actor_learner_threads[i].join()

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=0.7)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    K.set_session(sess)
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    else:
      env = GymEnvironment(config)

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    # Create a single instance of Agent to be multi-threaded
    agent = Agent(config, env, sess, threading.Lock())

    if FLAGS.is_train:
      init_threads(agent, config)
    else:
      agent.play(env)

if __name__ == '__main__':
  tf.app.run()
