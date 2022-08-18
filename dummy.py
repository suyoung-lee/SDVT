import metaworld
import random
'''
ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []


for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])

  print(env)
  print(task)

  env.set_task(task)
  training_envs.append(env)


for env in training_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action


'''

ml10 = metaworld.ML10()

train_env_name_list = [name for name, _ in ml10.train_classes.items()]
train_env_cls_list = [env_cls() for _, env_cls in ml10.train_classes.items()]
test_env_name_list = [name for name, _ in ml10.test_classes.items()]
test_env_cls_list = [env_cls() for _, env_cls in ml10.test_classes.items()]
train_tasks = ml10.train_tasks
test_tasks = ml10.test_tasks

print(train_env_name_list)
print(train_env_cls_list)

