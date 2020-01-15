import torch.nn.functional as F
import numpy as np
import torch
import gym
import argparse
import os
import sys
import datetime
import dateutil.tz

import utils
import TD3
import OurDDPG
import DDPG

from pygit2 import Repository


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False  # ??
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------", flush=True)
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}", flush=True)
    print("---------------------------------------", flush=True)
    return avg_reward


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE + KLD


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Pendulum-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e5, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--gr_save_freq", default=100, type=int)
    parser.add_argument("--max_timesteps", default=2e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=1500, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--vae_batch_size", type=int, default=100)
    args = parser.parse_args()
    log_dir = "./logs/" + Repository('.').head.shorthand

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    log_file = f"{args.policy}_{args.env}_{timestamp}.txt"

    # sys.stdout = open(os.path.join(log_dir, log_file), 'a+')

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------", flush=True)
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}", flush=True)
    print("---------------------------------------", flush=True)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models/GR")
        os.makedirs("./models/policy")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    max_action = float(action_high)
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    generative_replay = utils.JointGANTrainer(state_dim, action_dim, 100, 5, action_low, action_high, state_low,
                                              state_high)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]  # Used for evaluating abg reward after a certain time step

    state, done = env.reset(), False
    gr_train_count = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    train_batch = torch.zeros([args.vae_batch_size, 4], dtype=torch.float)
    gr_index = 0

    global_count = 0
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        vae_batch_size = args.vae_batch_size

        global_count = global_count + 1

        if gr_index >= vae_batch_size:
            gr_index = 0
            if t >= args.start_timesteps:
                vae_batch_size = int(args.vae_batch_size / 2)
                temp = generative_replay.sample(int(args.vae_batch_size / 2)).to(torch.device('cpu'))

                train_batch = torch.cat((train_batch, temp), 0)

            train_batch = generative_replay.normalise(train_batch)

            # Shuffle

            perm = torch.randperm(train_batch.size()[0])
            train_batch = train_batch[perm]

            d_loss, g_loss, gen_xy, gen_yx, critic, opt_g, opt_d = generative_replay.train(train_batch)

            print(
                "[Epoch %d] [D loss: %f] [G loss: %f]"
                % (global_count, d_loss, g_loss)
            )

            train_batch = torch.zeros([args.vae_batch_size, 4], dtype=torch.float)

            if global_count % 20 == 0:
                torch.save(gen_xy,
                           f"./models/GR/{Repository('.').head.shorthand}_{'G_XY'}_{args.env}.pth")

                torch.save(gen_yx,
                           f"./models/GR/{Repository('.').head.shorthand}_{'G_YX'}_{args.env}.pth")
                torch.save(opt_g,
                           f"./models/GR/{Repository('.').head.shorthand}_{'G'}_{args.env}_optimizer.pth")

                torch.save(critic,
                           f"./models/GR/{Repository('.').head.shorthand}_{'D'}_{args.env}.pth")
                torch.save(opt_d,
                           f"./models/GR/{Repository('.').head.shorthand}_{'D'}_{args.env}_optimizer.pth")

        train_batch[gr_index] = torch.FloatTensor(np.concatenate((state, action), 0))
        gr_index = gr_index + 1

        ## LOSS FUNCTION AND GRAD??

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(generative_replay, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}",
                flush=True)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/policy/{file_name}")

# sys.stdout.close()
