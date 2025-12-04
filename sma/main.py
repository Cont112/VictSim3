import random
import os

# importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

def main(vict_folder, env_folder, config_ag_folder):
    # Instantiate the environment
    env = Env(vict_folder, env_folder)

    # config files for the agents
    rescuer_file1 = os.path.join(config_ag_folder, "rescuer_1.txt")

    explorer_file1 = os.path.join(config_ag_folder, "explorer_1.txt")
    explorer_file2 = os.path.join(config_ag_folder, "explorer_2.txt")
    explorer_file3 = os.path.join(config_ag_folder, "explorer_3.txt")


    target_1 = (-100, -100) #NW
    target_2 = (100, -100)  #NE
    target_3 = (0, 100)     #S
    resc1 = Rescuer(env, rescuer_file1)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    Explorer(env, explorer_file1, resc1, target_1)
    Explorer(env, explorer_file2, resc1, target_2)
    Explorer(env, explorer_file3, resc1, target_3)
    # Run the environment simulator
    env.run()


if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")
    # dataset com sinais vitais das vitimas
    vict_folder = os.path.join("..", "datasets/vict/", "408v")

    # dataset do ambiente (paredes, posicao das vitimas)
    env_folder = os.path.join("..", "datasets/env/", "94x94_408v")

    # folder das configuracoes dos agentes
    curr = os.getcwd()
    config_ag_folder = os.path.join(curr, "config_ag_1")

    main(vict_folder, env_folder, config_ag_folder)
