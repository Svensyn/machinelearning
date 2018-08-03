import gym
import tensorflow as tf
import numpy as np
import os


class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()
        self.input_layer = tf.placeholder(dtype= tf.float32, shape= [None, state_size])

        #NeuralNetwork
        hidden_layer = tf.layers.dense(self.input_layer, 16, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_2 = tf.layers.dense(hidden_layer, 16, activation=tf.nn.relu, kernel_initializer=initializer)

        #output
        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)

        #probability and index
        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis = 1)

        #Rewards and Training
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)
        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        #list to hold placeholders
        self.gradients_to_apply = []

        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        optimizer = tf.train.AdamOptimizer(learning_rate= 1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    #adding rewards to discounted rewards numpy
    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    #normalizing rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards



#variables
discount_rate = 0.95
num_actions = 2
state_size = 4

#environment
env = gym.make("CartPole-v1")

path = "./cartpole-pg/"

#training variables
training_episodes = 5000
max_steps_per_episode = 10000
episode_batch_size = 100

agent = Agent(num_actions, state_size)
init = tf.global_variables_initializer()

#save model checkpoints
saver = tf.train.Saver(max_to_keep = 2)

if not os.path.exists(path):
    os.makedirs(path)



with tf.Session() as sess:

    sess.run(init)
    total_episode_rewards = []
    #Zeroing all gradients
    gradient_buffer = sess.run(tf.trainable_variables())
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for episode in range(training_episodes):

        #reset environment

        state = env.reset()
        episode_history = []
        episode_rewards = 0
        checked = False
        #stepping through environment
        for step in range(max_steps_per_episode):

            if episode % 100 ==0:
                env.render()

            #get weights for each action
            action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]})


            #taking action
            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])
            state_next, reward, done, _ = env.step(action_choice)

            #appending to episode history
            episode_history.append([state, action_choice, reward, state_next])

            state = state_next
            episode_rewards += reward

            #checking if done
            if done or step + 1 == max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                episode_history = np.array(episode_history)
                episode_history[:,2] = discount_normalize_rewards(episode_history[:,2])

                #getting gradients
                ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                    agent.actions: episode_history[:,1],
                                                                    agent.rewards: episode_history[:, 2]})
                #add gradients to buffer
                for index, gradient in enumerate(ep_gradients):
                    gradient_buffer[index] += gradient


                break
            if episode % episode_batch_size == 0:

                feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))
                sess.run(agent.update_gradients, feed_dict = feed_dict_gradients)
                #zero gradients once applied
                for index, gradient in enumerate(gradient_buffer):
                    gradient_buffer[index] = gradient * 0

            if episode % 100 == 0:
                saver.save(sess, path + "pg-checkpoint", episode)
                if not checked:
                    checked = True
                    print("Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))






#resetting graph
tf.reset_default_graph()
#checking spaces
"""print(env.observation_space)
print(env.action_space)

#max number of games to play
games_to_play = 10

#loop to play many games
for i in range(games_to_play):
    #resetting environment
    obs = env.reset()
    episode_rewards = 0
    done = False

    while not done:
        #rendering environment
        env.render()

        #do a random action
        action = env.action_space.sample()

        #Take a step with chosen action
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
    #print total rewards
    print(episode_rewards)"""
#close environment
env.close()