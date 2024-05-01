from gym.envs.registration import register

env_name = 'TurtleCPG'
module_name = __name__

register(
    id=env_name,
    entry_point=f'{module_name}:TurtleCPG',
)
