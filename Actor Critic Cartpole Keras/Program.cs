/*/
    Title: Actor Critic Method
    Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
    Date created: 2020/05/13
    Last modified: 2020/05/13
    Description: Implement Actor Critic Method in CartPole environment.
/*/
/*/
//// Introduction

This script shows an implementation of Actor Critic method on CartPole-V0 environment.

////// Actor Critic Method

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:

1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.

Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.

////// CartPole-V0

A pole is attached to a cart placed on a frictionless track. The agent has to apply
force to move the cart. It is rewarded for every time step the pole
remains upright. The agent, therefore, must learn to keep the pole from falling over.

////// References

- [CartPole](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
/*/

/*/
//// Setup
/*/
/*
import gym;
import numpy as np;
import tensorflow as tf;
from tensorflow import keras;
from tensorflow.keras import layers;
*/
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
//
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
//

namespace Actor_Critic_Cartpole_Keras
{
    class Program
    {
        private static object state;

        static void Main(string[] args)
        {
            // Configuration parameters for the whole setup
            var seed = 42;
            var gamma = 0.99;  // Discount factor for past rewards
            var max_steps_per_episode = 10000;
            // var env = gym.make("CartPole-v0");  // Create the environment
            CartPoleEnv env = new CartPoleEnv(WinFormEnvViewer.Factory);  // Create the environment
            env.Seed(seed);
            // var eps = np.finfo(np.float32).eps.item();  // Smallest number such that 1.0 + eps != 1.0
            var eps = 1e-5;  // Smallest number such that 1.0 + eps != 1.0

            /*/
            //// Implement Actor Critic network

            This network learns two functions:

            1. Actor: This takes as input the state of our environment and returns a
            probability value for each action in its action space.
            2. Critic: This takes as input the state of our environment and returns
            an estimate of total rewards in the future.

            In our implementation, they share the initial layer.
            /*/

            var num_inputs = 4;
            var num_actions = 2;
            var num_hidden = 128;

            LayersApi layers = new LayersApi();
            var inputs = layers.Input(shape: (num_inputs));
            var common = layers.Dense(num_hidden, activation: "relu").Apply(inputs);
            var action = layers.Dense(num_actions, activation: "softmax").Apply(common);
            var critic = layers.Dense(1).Apply(common);

            Model model = keras.Model(inputs: inputs, outputs: [action, critic]);

            /*/
            //// Train
            /*/

            var optimizer = keras.optimizers.Adam(learning_rate: (float)0.01);
            var huber_loss = keras.losses.Huber();
            var action_probs_history = new List<double>();
            var critic_value_history = new List<double>();
            var rewards_history = new List<double>();
            var running_reward = 0;
            var episode_count = 0;

            while (true)  // Run until solved
            {
                state = env.reset();
                var episode_reward = 0;
                using (var tape = tf.GradientTape())
                {
                    for (int timestep = 1; timestep < max_steps_per_episode; timestep++)
                    {
                        env.render(); // Adding this line would show the attempts
                                      // of the agent in a pop up window.

                        state = tf.convert_to_tensor(state);
                        state = tf.expand_dims(state, 0);

                        // Predict action probabilities and estimated future rewards
                        // from environment state
                        var (action_probs, critic_value) = model(state);
                        critic_value_history.Add(critic_value[0, 0]);

                        // Sample action from action probability distribution
                        action = np.random.choice(num_actions, p: np.squeeze(action_probs));
                        action_probs_history.Add(tf.math.log(action_probs[0, action]));

                        // Apply the sampled action in our environment
                        var (state, reward, done, _) = env.step(action);
                        rewards_history.Add(reward);
                        episode_reward += reward;

                        if (done)
                            break;
                    }
                    // Update running reward to check condition for solving
                    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward;

                    // Calculate expected value from rewards
                    // - At each timestep what was the total reward received after that timestep
                    // - Rewards in the past are discounted by multiplying them with gamma
                    // - These are the labels for our critic
                    var returns = new List<double>();
                    var discounted_sum = 0;

                    var reverse_rewards_history = rewards_history;
                    reverse_rewards_history.Reverse();
                    foreach (double r in reverse_rewards_history)
                    {
                        discounted_sum = r + gamma * discounted_sum;
                        returns.Insert(0, discounted_sum);
                    }

                    // Normalize
                    returns = np.array(returns);
                    returns = (returns - np.mean(returns)) / (np.std(returns) + eps);
                    returns = returns.ToList();

                    // Calculating loss values to update our network
                    var history = zip(action_probs_history, critic_value_history, returns);
                    var actor_losses = new List<double>();
                    var critic_losses = new List<double>();
                    foreach (var (log_prob, value, ret) in history)
                    {
                        // At this point in history, the critic estimated that we would get a
                        // total reward = `value` in the future. We took an action with log probability
                        // of `log_prob` and ended up recieving a total reward = `ret`.
                        // The actor must be updated so that it predicts an action that leads to
                        // high rewards (compared to critic's estimate) with high probability.
                        var diff = ret - value;
                        actor_losses.Add(-log_prob * diff);  // actor loss

                        // The critic must be updated so that it predicts a better estimate of
                        // the future rewards.
                        critic_losses.Add(
                            huber_loss.Call(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                        );
                    }

                    // Backpropagation
                    var loss_value = actor_losses.Sum(x => Convert.ToDouble(x)) + critic_losses.Sum(x => Convert.ToDouble(x));
                    var grads = tape.gradient(loss_value, model.trainable_variables);
                    optimizer.apply_gradients(zip(grads, model.trainable_variables));

                    // Clear the loss and reward history
                    action_probs_history.Clear();
                    critic_value_history.Clear();
                    rewards_history.Clear();

                }
                // Log details
                episode_count += 1;
                if (episode_count % 10 == 0)
                {
                    var template = String.Format("running reward: {0} at episode {1}", running_reward, episode_count);
                    Console.WriteLine(template);
                }

                if (running_reward > 195)  // Condition to consider the task solved
                {
                    Console.WriteLine(String.Format("Solved at episode {0}!", episode_count));
                    break;
                }
            }
            /*/
            //// Visualizations
            In early stages of training:
            ![Imgur](https://i.imgur.com/5gCs5kH.gif)

            In later stages of training:
            ![Imgur](https://i.imgur.com/5ziiZUD.gif)
            /*/

        }
    }
}
