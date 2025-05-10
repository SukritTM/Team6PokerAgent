import os, sys, time, inspect, types
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from poker_env import PokerEnv
from raise_player import RaisedPlayer
from PPOplayer import PPOPlayer

def plot_learning_curve(log_root: str, window: int = 500):
    csv_dir = os.path.join(log_root, "train")
    files   = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
               if f.startswith("monitor")]
    if not files:
        print("No monitor files yet.")
        return
    files.sort(key=os.path.getmtime)
    df = pd.read_csv(files[-1], skiprows=1)

    plt.figure(figsize=(10,4))
    plt.plot(df["r"].rolling(window).mean())
    plt.xlabel("Episode"); plt.ylabel(f"Reward (rolling {window})")
    plt.title("Poker PPO - learning curve")
    out = os.path.join(log_root, "learning_curve.png")
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"learning curve saved → {out}")

max_round = 500
# These should be the same I think
def make_train_env():
    return Monitor(PokerEnv(agent_model=PPOPlayer(), opponent_model=RaisedPlayer(), max_round=max_round), "./logs/train")

def make_eval_env():
    return Monitor(PokerEnv(agent_model=PPOPlayer(), opponent_model=RaisedPlayer(), max_round=max_round), "./logs/train")

# Followed example online -- seems to work, adjusted the learning rate a bit and the gae
def train_agent(total_timesteps=200000, eval_freq = 25000, save_freq = 25000, lr = 3e-4):

    for d in ("./logs/train", "./logs/eval", "./models/best"):
        os.makedirs(d, exist_ok=True)

    train_env = DummyVecEnv([make_train_env])
    eval_env  = DummyVecEnv([make_eval_env])

    policy_kwargs = dict(net_arch=[dict(pi=[256,256], vf=[256,256])])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        normalize_advantage=True, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        verbose=1, tensorboard_log="./logs/tb",
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models",
        name_prefix="ppo_poker",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best",
        log_path="./logs/eval",
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb], tb_log_name="ppo_poker", progress_bar=True)
    print(f"\nTraining finished in {(time.time()-t0)/60:.1f} min")

    model.save("./models/ppo_poker_final")
    plot_learning_curve("./logs")
    return model

if __name__ == "__main__":
    train_agent()
