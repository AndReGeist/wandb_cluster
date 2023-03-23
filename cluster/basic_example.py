"""
This is the same script as "basic_example.ipynb" but in a python script.
In addition, we used python-fire to efficiently create a command line interface.
"""

import time
import pickle
import os

import jax
import jax.nn as jnn
import equinox as eqx
import diffrax
import optax

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt

import fire
import wandb

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

def _get_data(ts, *, key):
    y0 = jrandom.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys

def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def create_dataset(seed=42, size_train=256, size_test=256):

    # ==== Initialize W&B run to track job ==== #
    config = {
        'size_train': size_train,
        'size_test': size_test,
        'seed': seed
    }
    run = wandb.init(project='wandb_cluster_neuralode', job_type='dataset-creation', config=config)

    key = jrandom.PRNGKey(seed)
    key_train, key_test = jrandom.split(key, 2)
    ts, ys = get_data(size_train, key=key_train)
    ts_test, ys_test = get_data(size_test, key=key_test)
    _, length_size, data_size = ys.shape

    data = {'ts': ts,
            'ys': ys,
            'length_size': length_size,
            'data_size': data_size,
            'ts_test': ts_test,
            'ys_test': ys_test}

    with open('data.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = wandb.Artifact('my-dataset', type='dataset')  # Create new artifact
    dataset.add_file('data.pickle')  # Add files to artifact
    run.log_artifact(dataset)  # Log artifact to save it as an output of this run

    wandb.finish()  # Finish session

def main(
        batch_size=32,
        lr_strategy=(3e-3, 3e-3),
        steps_strategy=(500, 500),
        length_strategy=(0.1, 1),
        width_size=64,
        depth=2,
        seed=5678,
        plot=False,
        print_every=100,
        watch_run = False
):

    # Define the config dictionary object
    config = {
        'batch_size': batch_size,
        'lr_strategy': lr_strategy,
        'steps_strategy': steps_strategy,
        'length_strategy': length_strategy,
        'width_size': width_size,
        'depth': depth,
        'seed': seed,
        'print_every': print_every
    }

    # ==== Initialize W&B run to track job ==== #
    run = wandb.init(
        project='wandb_cluster_neuralode',
        job_type='basic_example',
        config=config
    )
    # You can explicitly state to which team wandb will save data by adding the option entity='<Team name>'

    # When using sweep the default config gets overwritten
    config = wandb.config


    # ==== W&B - load data artifact ==== #
    artifact = run.use_artifact('my-dataset' + ':latest')
    artifact_dir = artifact.download()

    data_path = os.path.join(artifact_dir, 'data.pickle')
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    # ==== JAX - init model ==== #
    key = jrandom.PRNGKey(config.seed)
    model_key, loader_key = jrandom.split(key, 2)
    model = NeuralODE(data['data_size'], config.width_size, config.depth, key=model_key)


    # === Training loop === #
    # Until step 500 we train on only the first 10% of each time series.
    # This is a standard trick to avoid getting caught in a local minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    def pytree_leaves_to_dict(pytree_obj, name='array', pick_ndarrays=True):
        # W&B - To log the ndarrays of a JAX pytree, we need to extract the ndarrays,
        # transform them to numpy arrays and save these in a dict.
        # Rightnow, I did not figure out how to give these arrays an informative name
        leaves = jax.tree_util.tree_leaves(pytree_obj)
        for k, leaf in enumerate(leaves):
            if isinstance(leaf, jax.numpy.ndarray):
                key_name = name + str(k)
                log_dict[key_name] = np.array(leaf)
        return log_dict

    for lr, steps, length in zip(config.lr_strategy, config.steps_strategy, config.length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = data['ts'][: int(data['length_size'] * length)]
        _ys = data['ys'][:, : int(data['length_size'] * length)]
        for step, (yi,) in zip(
                range(steps), dataloader((_ys,), config.batch_size, key=loader_key)
        ):
            start = time.time()
            loss_train, model, opt_state, grads, updates = make_step(_ts, yi, model, opt_state)
            end = time.time()

            if (step % print_every) == 0 or step == steps - 1:
                print(f'Step: {step}, Loss: {loss_train}, Computation time: {end - start}')

                # Test model
                loss_test, _ = grad_loss(model, data['ts_test'], data['ys_test'])

                # === W&B - log optimization === #
                log_dict = {
                    'step': step,
                    'loss_train': loss_train,
                    'loss_test': loss_test,
                    'computation time': end - start
                }

                if watch_run:
                    log_dict.update( pytree_leaves_to_dict(model, name='model_array') )
                    #log_dict.update( pytree_leaves_to_dict(grads, name='grad_array') )
                    log_dict.update( pytree_leaves_to_dict(updates, name='updates_array') )

                wandb.log( log_dict )

    if plot:
        plt.plot(data['ts'], data['ys'][0, :, 0], c='dodgerblue', label='Real')
        plt.plot(data['ts'], data['ys'][0, :, 1], c='dodgerblue')
        model_y = model(data['ts'], data['ys'][0, 0])
        plt.plot(data['ts'], model_y[:, 0], c='crimson', label='Model')
        plt.plot(data['ts'], model_y[:, 1], c='crimson')
        plt.legend()
        plt.tight_layout()
        plt.savefig('neural_ode.png')
        plt.show()

    wandb.finish()  # Finish W&B session

if __name__ == '__main__':
    fire.Fire(main)
