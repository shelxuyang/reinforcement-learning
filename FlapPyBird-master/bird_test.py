import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'
ACTION = 2
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 2000000
FINAL_EPSILON = 0.0001
INITAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

def weight_varivable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding= "SAVE")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAVE")

def createNetwork():

    W_conv1 = weight_varivable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_varivable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_varivable([3, 3, 32, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_varivable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_varivable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    s = tf.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv2, 2) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1), b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):

    global readout_t, action_index
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.palceholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)

    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()

    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t, = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_RGB2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t,x_t,x_t,x_t), axis = 2)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_check_path)
        print("Successfully loaded", checkpoint.model_checkpoint_path)

    else:
        print("could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":

        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:

            if random.random() <= epsilon:
                print("---------------Random Action--------------------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1

            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

        else:
            a_t[0] = 1

    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITAL_EPSILON - FINAL_EPSILON) / EXPLORE

    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_RGB2GRAY)
    ret, x_t1 = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

    D.append((s_t, a_t, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    if t > OBSERVE:

        minibatch = random.sample(D, BATCH)

        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        y_batch = []
        readout_j1_batch = readout.eval(feed_dict = {s : s_j_batch})
        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]

            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

        train_step.run(feed_dict = {
            y: y_batch,
            a: a_batch,
            s: s_j_batch}
        )

    s_t = s_t1
    t += 1

    if t% 10000 == 0:
        saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=4)

    state = ""

    if  t <= OBSERVE:
        state = "observe"
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"


    print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

if __name__ == '__main__':
    playGame()