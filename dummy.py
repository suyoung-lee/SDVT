import metaworld
import random


ml10 = metaworld.ML10()

train_env_name_list = [name for name, _ in ml10.train_classes.items()]
train_env_cls_list = [env_cls() for _, env_cls in ml10.train_classes.items()]
test_env_name_list = [name for name, _ in ml10.test_classes.items()]
test_env_cls_list = [env_cls() for _, env_cls in ml10.test_classes.items()]
train_tasks = ml10.train_tasks
test_tasks = ml10.test_tasks

print(train_env_name_list)
print(train_env_cls_list)

