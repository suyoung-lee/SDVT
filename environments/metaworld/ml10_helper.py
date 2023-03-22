import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []



def reset_task():
    env_ind = random.choice(range(10))
    _env = [env_cls() for _,env_cls in ml10.train_classes.items()][env_ind]
    _env_name = [name for name, _ in ml10.train_classes.items()][env_ind]
    _task = random.choice([_task for _task in ml10.train_tasks
                           if _task.env_name == _env_name])
    print(_env)
    print(_task)

    _env.set_task(_task)
    return _env

_env = reset_task()
print(_env.action_space)
print(_env.observation_space)