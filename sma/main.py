import os

# importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer


def main(vict_folder, env_folder, config_ag_folder):
    # Instantiate the environment
    env = Env(vict_folder, env_folder)

    # config files for the agents
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1.txt")
    explorer_file = os.path.join(config_ag_folder, "explorer_1.txt")

    # Instantiate agents rescuer and explorer
    resc = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    Explorer(env, explorer_file, resc)

    # Run the environment simulator
    env.run()


if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")
    # dataset com sinais vitais das vitimas
    vict_folder = os.path.join("..", "datasets/vict/", "10v")

    # dataset do ambiente (paredes, posicao das vitimas)
    env_folder = os.path.join("..", "datasets/env/", "12x12_10v")

    # folder das configuracoes dos agentes
    curr = os.getcwd()
    config_ag_folder = os.path.join(curr, "config_ag_1")

    main(vict_folder, env_folder, config_ag_folder)
