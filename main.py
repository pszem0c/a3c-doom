import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import skimage
from skimage import transform

from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time
import os

def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
  import moviepy.editor as mpy
  
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  
  def make_mask(t):
    try:
      x = salIMGS[int(len(salIMGS)/duration*t)]
    except:
      x = salIMGS[-1]
    return x

  clip = mpy.VideoClip(make_frame, duration=duration)
  if salience == True:
    mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
    clipB = clip.set_mask(mask)
    clipB = clip.set_opacity(0)
    mask = mask.set_opacity(0.1)
    mask.write_gif(fname, fps = len(images) / duration,verbose=False)
    #clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
  else:
    clip.write_gif(fname, fps = len(images) / duration,verbose=False)

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(frame):
    s = frame[10:-10, 30:-30]
    s = transform.resize(s,[84,84])
    s = np.reshape(s, [np.prod(s.shape)])/255.0
    return s

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::1], axis=0)[::-1]

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std/np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class ACNetwork():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                    inputs=self.imageIn, 
                    num_outputs=16,
                    kernel_size=[8,8],
                    stride=[4,4],
                    weights_initializer='random_uniform',
                    padding='VALID')

            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                    inputs=self.conv1, 
                    num_outputs=32,
                    kernel_size=[4,4],
                    stride=[2,2],
                    weights_initializer='random_uniform',
                    padding='VALID')

            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu,
                    weights_initializer='random_uniform')

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init  = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in,h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                    time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            self.policy = slim.fully_connected(rnn_out,
                    a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,
                    1,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)

            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = (tf.reduce_sum(self.policy*self.actions_onehot, [1]))
                
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy*tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss  =0.5*self.value_loss + self.policy_loss - self.entropy * 0.01

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        self.localac = ACNetwork(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('gobal', self.name)
        
        game.set_doom_scenario_path("basic.wad")
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()

        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma*self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages , gamma)

        feed_dict = {self.localac.target_v: discounted_rewards,
                self.localac.inputs: np.vstack(observations),
                self.localac.actions: actions,
                self.localac.advantages: advantages,
                self.localac.state_in[0]: self.batch_rnn_state[0],
                self.localac.state_in[1]: self.batch_rnn_state[1]}
        value_loss, policy_loss, entropy, grad_norms, var_norms, self.batch_rnn_state, _ = sess.run(
                [self.localac.value_loss,
                    self.localac.policy_loss,
                    self.localac.entropy,
                    self.localac.grad_norms,
                    self.localac.var_norms,
                    self.localac.state_out,
                    self.localac.apply_grads],
                feed_dict = feed_dict) 
        return value_loss / len(rollout), policy_loss / len(rollout), entropy / len(rollout), grad_norms, var_norms

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.localac.state_init
                self.batch_rnn_state = rnn_state
                while self.env.is_episode_finished() == False:
                    a_dist, v, rnn_state = sess.run([self.localac.policy, self.localac.value, self.localac.state_out],
                            feed_dict = {self.localac.inputs: [s],
                                self.localac.state_in[0]: rnn_state[0],
                                self.localac.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.make_action(self.actions[a]) / 100.0
                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else: 
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        v1 = sess.run(self.localac.value,
                                feed_dict = {self.localac.inputs: [s],
                                    self.localac.state_in[0]: rnn_state[0],
                                    self.localac.state_in[1]: rnn_state[1]})[0, 0]
                        value_loss, policy_loss, entropy, grad_norms, var_norms = self.train(episode_buffer,
                                sess,
                                gamma,
                                v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    value_loss, policy_loss, entropy, grad_norms, var_norms = self.train(episode_buffer, 
                            sess,
                            gamma,
                            0.0)

                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image'+str(episode_count)+'.gif',
                                duration=len(images)*time_per_step, true_image=True, salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(value_loss))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(entropy))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

max_episode_length = 300
gamma = 0.99
s_size = 7056
a_size = 3
load_model = False
model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=0.1)
    master_network = ACNetwork(s_size, a_size, 'global', None)
    num_workers = multiprocessing.cpu_count()
    workers = []

    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print("Loading model...")
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
