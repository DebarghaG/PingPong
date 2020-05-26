# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import pong_game
import numpy as np
import random
import os
from collections import deque
from numpy.random import choice
import time
from pygame import *
import pygame

"""
To store the image data, that's to be fed to the frame uncomment this line
from PIL import Image
"""


"""
It has the following hyper-parameters :
1. Action Space ​ of 3, which can be (no movement, go up, or go down)
2. Gamma : Describes the learning rate
3. Initial Epsilon : ​ The epsilon value we start from (1)
4. Final Epsilon : ​ The epsilon value we end at (0.05)
5. Observation : ​ The initial frames for which the agent will just observe.
6. Exploration : ​ The frames for which the greedy epsilon strategy will reduce the epsilon
from the initial epsilon value to the final epsilon value.
7. Memory Size : ​ The amount of memory provided to the agent to keep track of it’s
previous experience. ( Experience replay )
8. Batch Size : ​ The number of frames over which training takes place. (48)
"""
class Controller:
    def __init__(self):
        self.actions_space  = 3
        self.gamma          = 0.99
        self.eps_i          = 1.0
        self.eps_f          = 0.05
        self.exploration    = 10000
        self.observation    = 1000
        self.predict        = True
        self.checkpoint_here= 5000
        self.replay_mem_size= 200000
        self.batch_size     = 48

    """
    Network Architecture :
    For feature engineering :
        1. 1st Convolutional Layer​ : Takes a 60x60 frame as input.
        2. Maxpooling Layer​ : Reduces dimensionality of feature map
        3. 2nd Convolutional Layer​ : More feature engineering
        4. 3rd Convolutional Layer ​ : More feature engineering

    The result of this feature engineering is flattened into a tensor.

    This is followed by :
        1. Fully connected 4th Layer ​ : Accepts flattened tensor
        2. Fully connected 5th Layer ​ : Outputs a tensor of the size of Action Space

    The result of the fully connected 5th layer is the network output. ReLu activation functions have
    been used throughout.
    """
    def Create_DQNetwork(self):
         #with tf.device('/gpu:0')
         #If you have a GPU device or using collab.

         with tf.device('/cpu:0'):
            conv1_weights                       = tf.Variable(tf.random.truncated_normal([6, 6, 4, 32], stddev=0.02))
            conv1_biases                        = tf.Variable(tf.constant(0.01, shape=[32]))

            conv2_weights                       = tf.Variable(tf.random.truncated_normal([4, 4, 32, 64], stddev=0.02))
            conv2_biases                        = tf.Variable(tf.constant(0.01, shape=[64]))

            conv3_weights                       = tf.Variable(tf.random.truncated_normal([3, 3, 64, 64], stddev=0.02))
            conv3_biases                        = tf.Variable(tf.constant(0.01, shape=[64]))

            fullyconnected_layer4_weights       = tf.Variable(tf.random.truncated_normal([1024, 512], stddev=0.02))
            fullyconnected_layer4_biases        = tf.Variable(tf.constant(0.01, shape=[512]))

            fullyconnected_layer5_weights       = tf.Variable(tf.random.truncated_normal([512, self.actions_space], stddev=0.02))
            fullyconnected_layer5_biases        = tf.Variable(tf.constant(0.01, shape=[self.actions_space]))


            pixel_input                         = tf.placeholder("float", [None, 50, 60, 4])

            convolution1_output                 = tf.nn.relu(tf.nn.conv2d(pixel_input, conv1_weights, strides = [1, 4, 4, 1], padding = "SAME") + conv1_biases)
            maxpooling                          = tf.nn.max_pool(convolution1_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            convolution2_output                 = tf.nn.relu(tf.nn.conv2d(maxpooling, conv2_weights, strides = [1, 2, 2, 1], padding = "SAME") + conv2_biases)
            convolution3_output                 = tf.nn.relu(tf.nn.conv2d(convolution2_output, conv3_weights, strides = [1, 1, 1, 1], padding = "SAME") + conv3_biases)
            flattened_conv3_output              = tf.reshape(convolution3_output, [-1, 1024])
            fullyconnected_layer4_output        = tf.nn.relu(tf.matmul(flattened_conv3_output, fullyconnected_layer4_weights) + fullyconnected_layer4_biases)
            fullyconnected_layer5_output        = tf.matmul(fullyconnected_layer4_output, fullyconnected_layer5_weights) + fullyconnected_layer5_biases


            print("Printing the dimensions of the layers : ")

            print(tf.shape(convolution1_output))
            print(tf.shape(maxpooling))
            print(tf.shape(convolution2_output))
            print(tf.shape(convolution3_output))
            print(tf.shape(flattened_conv3_output))

            #time.sleep(0)
            return pixel_input, fullyconnected_layer5_output

    """
    Deep Q Learning with Experience Replay Pseudocode:


    1. Initialising the replay memory D to it's capacity N.
    2. Initialising the action-value Q function with random weights
    3. Iterate through the Episodes (I've created a continuous environment, this step in unnecessary)
        4. Initialise the state at the beginning, and the reward of the first frame.
        5. Iterate through the timesteps that the Model needs to be trained for.
            6. Using greedy epsilon strategy choose an action for the step t or from action tensor
            7. Execute the option and observe the reward, and get the next frame
            8. Move to the next state, initialise the experience replay queue to contain (stacked_frames, argmax_tensor(action), reward_tensor, local_stacked_frames)
            9. Sample random minibatch of transitions from experience replay experience_replay_queue
            10.Set the reality value according the gamma, reward recieved and the Q value of the frame.
            11.Gradient descent step on the predicted Q value and the reality
        12. Iterate
    13. Iterate
    """
    def trainGraph(self,input_pixels, network_output):
        action_argmax       = tf.placeholder("float", [None, self.actions_space])
        reality             = tf.placeholder("float", [None])
        step_number         = tf.Variable(0, name='step_number')

        action              = tf.reduce_sum(tf.multiply(network_output, action_argmax), reduction_indices = 1)
        cost_for_action     = tf.reduce_mean(tf.square(action - reality))
        train_step          = tf.train.AdamOptimizer(1e-6).minimize(cost_for_action)

        game = pong_game.PongGame()
        experience_replay_queue = deque()

        image_frame                 = game.getPresentFrame()
        intermediate_frame          = cv2.resize(image_frame, (60, 60))
        image_frame                 = cv2.cvtColor(intermediate_frame[0:50,:], cv2.COLOR_BGR2GRAY)
        returned_val, image_frame   = cv2.threshold(image_frame, 1, 255, cv2.THRESH_BINARY)
        stacked_frames              = np.stack((image_frame, image_frame, image_frame, image_frame), axis = 2)

        """
        Uncomment to capture the state of the environment
        #img = Image.fromarray(image_frame)
        #img.save('first.png')
        #stack frames, that is our input tensor
        """

        saver        = tf.train.Saver(tf.global_variables())
        session      = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        checkpoint   = tf.train.latest_checkpoint('./checkpoints')

        if checkpoint is not None:
            print('Checkpoint found at the timestep %s'%(checkpoint))
            saver.restore(session, checkpoint)
            print("Model restored.")

        else:
            init        = tf.global_variables_initializer()
            session.run(init)
            print("Initialized new Graph")

        timestep        = step_number.eval()
        counter         = 0
        local_epsilon   = self.eps_i
        global_score    = 0
        global_game     = 0

        """
        TO PLAY AGAINST A HUMAN :


        while(True):

            keys=pygame.key.get_pressed()
            if keys[K_UP]:
                HAction = 1
            elif keys[K_DOWN]:
                HAction = 2
            else:
                HAction = 0
        """
        while(True):
            Q_Table          = network_output.eval(feed_dict = {input_pixels : [stacked_frames]})[0]
            argmax_tensor    = np.zeros([self.actions_space])

            if(random.random() <= local_epsilon and not self.predict):
                actionIndex = choice((0,1,2), 1, p=(0.90, 0.05,0.05))
            else:
                actionIndex = np.argmax(Q_Table)
            argmax_tensor[actionIndex] = 1

            if local_epsilon > self.eps_f:
                local_epsilon = local_epsilon - (self.eps_i - self.eps_f) / self.exploration

            Phase = 'Agent is Observing'
            if timestep > self.observation:
                Phase = 'Agent is training'
            if self.predict:
                Phase = 'Agent is Playing'

            """
            #print(timestep)            ==> Timestep
            #print(argmax_tensor)       ==> Tensor who's argmax is the action
            #print(np.max(Q_Table))     ==> The Maximum Q Value, i.e. the Q Value of the prediction
            #print(local_epsilon)       ==> Tells how far the greedy epsilon strategy has come
            #print(Phase)
            """

            reward_tensor, image_frame = game.getNextFrame(argmax_tensor, [timestep, np.max(Q_Table), local_epsilon, Phase])

            intermediate_frame         = cv2.resize(image_frame, (60, 60))
            image_frame                = cv2.cvtColor(intermediate_frame[0:50,:], cv2.COLOR_BGR2GRAY)
            returned_val, image_frame  = cv2.threshold(image_frame, 1, 255, cv2.THRESH_BINARY)

            """
            In case snapshots of the frame need to be taken during the Training Frame.

            #img = Image.fromarray(image_frame)
            #time.sleep(100)
            #img.save('train.png')
            #img.show()
            """
            image_frame                = np.reshape(image_frame, (50, 60, 1))
            local_stacked_frames       = np.append(image_frame, stacked_frames[:, :, 0:3], axis = 2)

            """
            #print(stacked_frames)              ==> Stacked the frames
            #print(argmax_tensor)               ==> Tensor who's argmax is the action
            #print(reward_tensor)               ==> Reward recieved
            #print(local_stacked_frames)        ==> Frame t+1 that has been retrieved

            #time.sleep(10)
            """

            experience_replay_queue.append((stacked_frames, argmax_tensor, reward_tensor, local_stacked_frames))
            if len(experience_replay_queue) > self.replay_mem_size:
                experience_replay_queue.popleft()


            if counter > self.observation and not self.predict:


                minibatch = random.sample(experience_replay_queue, self.batch_size)
                """
                minibatch => Random sample of batch size from the experience replay D
                """

                input_batch_x       = [temp_batch[0] for temp_batch in minibatch]
                argmax_batch_x      = [temp_batch[1] for temp_batch in minibatch]
                reward_batch_x      = [temp_batch[2] for temp_batch in minibatch]
                inp_t1_batch_x      = [temp_batch[3] for temp_batch in minibatch]

                """
                input_batch_x       => State of the simulator at timestep t (stacked_frames)
                argmax_batch_x      => Action taken at timestep t (Argmax_tensor)
                reward_batch_x      => Reward Tensor recieved at time t
                inp_t1_batch_x      => The input of the next frame
                """

                reality_batch       = []
                out_batch           = network_output.eval(feed_dict = {input_pixels : inp_t1_batch_x})


                """
                Reward for the
                """
                for j in range(0, len(minibatch)):
                    reality_batch.append(reward_batch_x[j] + self.gamma * np.max(out_batch[j]))

                train_step.run(feed_dict = {reality : reality_batch, action_argmax : argmax_batch_x, input_pixels : input_batch_x})

            stacked_frames  = local_stacked_frames
            timestep        = timestep + 1
            counter         = counter + 1

            """
            stacked_frames  => Setting the frames of t to t+1
            timestep        => Incrementing the timestep of the learning process
            counter         => Counter that's to be incremented
            """

            if timestep % self.checkpoint_here == 0:
                if not self.predict:
                    session.run(step_number.assign(timestep))
                    saver.save(session, './checkpoints/model.ckpt', global_step=timestep)

            if reward_tensor != 0:
                """
                Auxilary Reward Tracing

                    if reward_tensor==1:
                        if global_score % 21 ==0:
                            global_game = global_game + 1
                            global_score= 0

                    if reward_tensor==-1:
                        if global_score % 21 ==0:
                            global_game = global_game - 1
                            global_score= 0

                    print("Agent has played : {} on game : {} ".format(global_score, global_game))
                """
                log   = open("log.csv","a")
                print("Phase:{} --> Timestep:{} ,Epsilon:{} ,Action:{} ,Reward:{} ,Q Value of Action:{}".format(Phase,timestep,local_epsilon,actionIndex,reward_tensor,np.max(Q_Table)))
                log.write("Timestep:{},Epsilon:{},Action:{},Reward:{},Q Value of Action:{}\n".format(timestep,local_epsilon,actionIndex,reward_tensor,np.max(Q_Table)))
                log.close()


"""
Main function does primarily :
1. Creates the checkpoint directory if it doesn't exist
2. Creates an Agent object, that corresponds to the Controller.
3. Creates the Deep Q Network, and the placeholder containing the image size
4. Calls the method that conducts the training of the network.
"""
def main():
    if not os.path.exists('./checkpoints'):
        print("Creating the checkpoint directory")
        os.makedirs('./checkpoints')
    agent = Controller()
    print("Initialised the controller")
    input_pixels, network_output = agent.Create_DQNetwork()
    print("The network has been created with initial weights")
    agent.trainGraph(input_pixels, network_output)
    print("Training objective has been completed")

if __name__ == "__main__":
    main()
