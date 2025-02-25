import numpy as np
import pickle, os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

def avg_return_curve(x, y, stride, total_steps):
    """
    Author: Rupam Mahmood (armahmood@ualberta.ca)
    Computes the average episodic return over timesteps.
    :param x: List (over runs) of lists of termination steps.
    :param y: List (over runs) of lists of episodic returns.
    :param stride: Timestep interval between aggregated datapoints.
    :param total_steps: Total number of timesteps to consider.
    :return: steps, average returns, standard errors.
    """
    assert len(x) == len(y)
    num_runs = len(x)
    num_points = total_steps // stride
    avg_ret = np.zeros(num_points)
    stderr_ret = np.zeros(num_points)
    steps = np.arange(stride, total_steps + stride, stride)
    for i in range(num_points):
        avg_rets_per_run = []
        for run in range(num_runs):
            xa = np.array(x[run])
            ya = np.array(y[run])
            mask = np.logical_and(i * stride < xa, xa <= (i + 1) * stride)
            if np.any(mask):
                avg_rets_per_run.append(np.mean(ya[mask]))
        if avg_rets_per_run:
            avg_ret[i] = np.mean(avg_rets_per_run)
            stderr_ret[i] = np.std(avg_rets_per_run) / np.sqrt(len(avg_rets_per_run))
        else:
            avg_ret[i] = np.nan
            stderr_ret[i] = np.nan
    return steps, avg_ret, stderr_ret

def load_experiment_data(data_dir):
    all_termination_time_steps, all_episodic_returns = [], []
    env_name = ""
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(data_dir, file), "rb") as f:
                episodic_returns, termination_time_steps, env_name = pickle.load(f)
                all_termination_time_steps.append(termination_time_steps)
                all_episodic_returns.append(episodic_returns)
    return all_termination_time_steps, all_episodic_returns, env_name

def load_final_returns(data_dir, total_steps=500_000):
    final_returns = []
    env_name = None
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(data_dir, file), "rb") as f:
                episodic_returns, termination_time_steps, env_name = pickle.load(f)
            run_returns = [ret for ts, ret in zip(termination_time_steps, episodic_returns) if ts <= total_steps]
            if run_returns:
                final_return = run_returns[-1]
            else:
                final_return = episodic_returns[-1]
            final_returns.append(final_return)
    return final_returns, env_name

def plot_reward_curves(data_dirs, labels, int_space, total_steps, output_file):
    plt.figure(figsize=(8, 5))
    env_name_global = None
    colors = ['orange', 'blue', 'green', 'red']
    for idx, data_dir in enumerate(data_dirs):
        x, y, env_name = load_experiment_data(data_dir.strip())
        if env_name_global is None:
            env_name_global = env_name
        steps, avg_ret, stderr_ret = avg_return_curve(x, y, int_space, total_steps)
        color = colors[idx % len(colors)]
        plt.fill_between(steps, avg_ret - stderr_ret, avg_ret + stderr_ret, color=color, alpha=0.2)
        plt.plot(steps, avg_ret, linewidth=2.0, color=color, label=labels[idx])
    
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Average Episodic Return", fontsize=14)
    plt.title(f"Reward Curves in {env_name_global}", fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(0, -0.25), ncol=2)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_bar_plots(data_dirs, labels, total_steps, output_file):
    results = []
    env_name_global = None
    for idx, data_dir in enumerate(data_dirs):
        final_returns, env_name = load_final_returns(data_dir.strip(), total_steps=total_steps)
        if env_name_global is None:
            env_name_global = env_name
        for ret in final_returns:
            results.append({"Algorithm": labels[idx], "FinalReturn": ret})
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(8, 5))
    custom_colors = ['orange', 'blue', 'green', 'red']
    ax = sns.barplot(x="Algorithm", y="FinalReturn", data=results_df, palette=custom_colors, ci=None)
    
    max_return = results_df["FinalReturn"].max()
    ax.set_ylim(0, max_return * 1.15)
    
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        offset = max_return * 0.03
        ax.text(patch.get_x() + patch.get_width() / 2, height + offset,
                f'{height:.0f}', ha='center', va='bottom', fontsize=12, color='black')
    
    plt.xlabel("Algorithm", fontsize=14)
    plt.ylabel("Final Average Episodic Return", fontsize=14)
    plt.title(f"Final Performance in {env_name_global}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def main(args):
    data_dirs = args.data_dirs.split(',')
    labels = args.labels.split(',')
    plot_reward_curves(data_dirs, labels, args.int_space, args.total_steps, args.output_file_rc)
    plot_bar_plots(data_dirs, labels, args.total_steps, args.output_file_bp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', type=str, default="""data_classic_ac_CartPole-v1_lr0.001_gamma0.99,
                                                            data_stream_ac_CartPole-v1_lr1.0_gamma0.99_lamda0.8_entropy_coeff0.1,
                                                            data_stream_ac_CartPole-v1_lr1.0_gamma0.99_lamda0.8_entropy_coeff0.01,
                                                            data_stream_ac_CartPole-v1_lr1.0_gamma0.99_lamda0.8_entropy_coeff0.5""")
    parser.add_argument('--labels', type=str, default='Classic AC,Stream AC τ=0.1,Stream AC τ=0.01,Stream AC τ=0.5')
    parser.add_argument('--int_space', type=int, default=5000)
    parser.add_argument('--total_steps', type=int, default=500000)
    parser.add_argument('--output_file_rc', type=str, default='all_reward_curves.pdf')
    parser.add_argument('--output_file_bp', type=str, default='final_avg_return_barplot.pdf')
    args = parser.parse_args()
    
    main(args)
