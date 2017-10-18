from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, conv2d, clipped_error
from .utils import get_time, save_pkl, load_pkl

class Agent(BaseModel):
  def __init__(self, config, env,  sess, lock):
    super(Agent, self).__init__(config)
    self.env = env
    self.sess = sess
    self.weight_dir = 'weights'
    self.start_step = None
    self.lock = lock

    self.batch_size = config.batch_size
    self.screen_height = config.screen_height
    self.screen_width = config.screen_width

    self.min_reward = config.min_reward
    self.max_reward = config.max_reward

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def init_states(self):
    actions = np.empty(self.batch_size, dtype=np.uint8)
    rewards = np.empty(self.batch_size, dtype=np.int)
    prestates = np.empty((self.batch_size, self.history_length, self.screen_height, self.screen_width), dtype=np.float16)
    poststates = np.empty((self.batch_size, self.history_length, self.screen_height, self.screen_width), dtype=np.float16)
    terminals = np.empty(self.batch_size, dtype=np.bool)

    if self.cnn_format == 'NHWC':
      prestates = np.empty((self.batch_size, self.screen_height, self.screen_width, self.history_length), dtype=np.float16)
      poststates = np.empty((self.batch_size, self.screen_height, self.screen_width, self.history_length), dtype=np.float16)

    return actions, rewards, prestates, poststates, terminals

  def train(self, environment, thread_id):
    # Initialize history
    history = History(self.config)

    # Initialize start step
    self.start_step = self.step_op.eval(session=self.sess)

    # Initialize separate environment for each agent thread
    env = environment
    start_time = time.time()

    # initialize variables
    num_game, update_count, ep_reward = 0, 0, 0.
    total_reward, total_loss, total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actionsList = [], []
    actions, rewards, prestates, poststates, terminals = self.init_states()
    current = 0

    time.sleep(thread_id % 10)

    screen, reward, action, terminal = env.new_random_game(self.lock)

    # stack 'history_length' frames to be given as input to the cnn
    for _ in range(self.history_length):
      history.add(screen)

    # repeat for max_steps
    t = 0
    epsilon = 1.0
    final_epsilon = self.get_final_epilon()
    print("Starting thread ", thread_id, "with final epsilon ", final_epsilon, end="\n")
    for self.step in tqdm(range(self.start_step, self.max_step), ncols=70, initial=self.start_step):

      # get current stacked state
      s_t = history.get()
      prestates[current, ...] = s_t

      # 1. predict
      action, epsilon = self.predict(s_t, env, epsilon=epsilon, final_epsilon=final_epsilon)

      # 2. act
      screen, reward, terminal = env.act(action, self.lock, is_training=True)
      history.add(screen)

      reward = max(self.min_reward, min(self.max_reward, reward))
      actions[current] = action
      rewards[current] = reward
      poststates[current, ...] = history.get()
      terminals[current] = terminal

      t += 1

      # 3. observe
      # update target network parameters
      if (self.step+1) % self.target_q_update_step == 0:
        self.update_target_q_network()

      # perform gradient descent
      if t % self.train_frequency == 0 or terminal:
        loss, q_t_mean = self.q_learning_mini_batch(prestates, poststates, actions, rewards, terminals, current)
        total_loss += loss
        total_q += q_t_mean
        update_count += 1

        # clear content
        actions, rewards, prestates, poststates, terminals = self.init_states()
        current = -1

      if terminal:
        screen, reward, action, terminal = env.new_random_game(self.lock)

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
        current = 0
      else:
        ep_reward += reward
        current = (current + 1) % self.batch_size

      actionsList.append(action)
      total_reward += reward

      if t >= self.learn_start:
        if self.step % self.test_step == 0:
          avg_reward = total_reward / self.test_step
          avg_loss = total_loss / update_count
          avg_q = total_q / update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d, Step #: %d' \
                % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game, self.step))

          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval(session=self.sess, feed_dict={self.step_input: self.step + 1})
            self.save_model()

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actionsList,
                'training.learning_rate': self.learning_rate_op.eval(session=self.sess,feed_dict={self.learning_rate_step: self.step}),
              }, self.step)

          num_game = 0
          total_reward = 0.
          total_loss = 0.
          total_q = 0.
          update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actionsList = []

  def predict(self, s_t, env, test_ep=None, epsilon=1., final_epsilon=0.1):
    epsilon = test_ep or epsilon

    if random.random() <= epsilon:
      action = random.randrange(env.action_size)
    else:
      action = self.q_action.eval(session=self.sess, feed_dict={self.s_t: [s_t]})[0]

    # scale down epsilon
    if epsilon > final_epsilon:
      epsilon -= (self.initial_epsilon - final_epsilon) / self.anneal_epsilon_timesteps

    return action, epsilon

  def q_learning_mini_batch(self, s_t, s_t_plus_1, action, reward, terminal, current):

    s_t = s_t[:current+1, ...]
    s_t_plus_1 = s_t_plus_1[:current+1, ...]
    action = action[:current+1]
    reward = reward[:current+1]
    terminal = terminal[:current+1]

    t = time.time()
    if self.double_q:
      # Double Q-learning
      pred_action = self.q_action.eval(session=self.sess,feed_dict={self.s_t: s_t_plus_1})

      q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
      })
      target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
    else:
      q_t_plus_1 = self.target_q.eval(session=self.sess, feed_dict={self.target_s_t: s_t_plus_1})

      terminal = np.array(terminal) + 0.
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss, summary_str  = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
    })

    self.writer.add_summary(summary_str, self.step)
    return loss, q_t.mean()

  def get_final_epilon(self):
    """http://arxiv.org/pdf/1602.01783v1.pdf"""
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
      else: # (?,4,32,32)
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')

      # get the q action
      self.q_action = tf.argmax(self.q, dimension=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l4, self.env.action_size, name='target_q')

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval(session=self.sess, feed_dict={self.t_w_input[name]: self.w[name].eval(session=self.sess)})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, step)

  def play(self, env, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    # if not self.display:
    #   gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
    #   self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = env.new_random_game(self.lock)
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action, _ = self.predict(test_history.get(), env, test_ep=test_ep)
        # 2. act
        screen, reward, terminal = env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print("="*30)
      print(" [%d] Best reward : %d" % (best_idx, best_reward))
      print("="*30)

    if not self.display:
      self.env.env.monitor.close()