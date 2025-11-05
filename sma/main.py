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
    rescuer_file2 = os.path.join(config_ag_folder, "rescuer_2.txt")
    rescuer_file3 = os.path.join(config_ag_folder, "rescuer_3.txt")

    explorer_file1 = os.path.join(config_ag_folder, "explorer_1.txt")
    explorer_file2 = os.path.join(config_ag_folder, "explorer_2.txt")
    explorer_file3 = os.path.join(config_ag_folder, "explorer_3.txt")

    base_order = [0, 2, 4, 6, 1, 3, 5, 7] # Prioriza (Cima, Direita, Baixo, Esquerda)
    #random.shuffle(base_order)
    # Agente 1: Ordem padrão
    order_1 = base_order
        
    # Agente 2: Rotaciona a lista, prioriza (Direita, Baixo, ...)
    order_2 = base_order[1:] + base_order[:1] 
    # Fica: [2, 4, 6, 1, 3, 5, 7, 0]

    # Agente 3: Rotaciona a lista, prioriza (Baixo, Esquerda, ...)
    order_3 = base_order[2:] + base_order[:2]
    # Fica: [4, 6, 1, 3, 5, 7, 0, 2]

    # Instantiate agents rescuer and explorer
    resc1 = Rescuer(env, rescuer_file1)
    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    Explorer(env, explorer_file1, resc1, order_1)
    Explorer(env, explorer_file2, resc1, order_2)
    Explorer(env, explorer_file3, resc1, order_3)
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
