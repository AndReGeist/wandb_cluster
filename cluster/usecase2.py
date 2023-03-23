from basic_example import main as main_basicexample
import fire
import wandb

def main():
    # Define the search space
    # You can specify...
    # a range 'x': {'max': 0.1, 'min': 0.01},
    # or values 'y': {'values': [1, 3, 7]},
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'loss_test'},
        'parameters':
            {
                'batch_size': {'values': [32]}, # 32
                'lr_strategy': {'values': [(3e-3, 3e-3)]}, # (3e-3, 3e-3)
                'steps_strategy': {'values': [(500, 500)]}, # (500, 500)
                'length_strategy': {'values': [(0.1, 1)]}, # (0.1, 1)
                'width_size': {'values': [20, 64, 150]}, # 64
                'depth': {'values': [1, 2, 3]}, # 2
                'seed': {'values': [42]},
                'print_every': {'values': [100]}  # 100
            }
    }

    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='wandb_cluster_neuralode')
    wandb.agent(sweep_id, function=main_basicexample, count=3)

if __name__ == '__main__':
    fire.Fire(main)