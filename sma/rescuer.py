##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from joblib import load
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import heapq

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None             # explorer will pass the map
        self.victims = None         # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan

        self.classifier = load('classifier.joblib')  # load the pre-trained classifier
        self.cluster_data = []

        self.num_explorers = 3
        self.collected_maps = []
        self.collected_victims = []
        self.unified_map = Map()
        self.unified_victims = {}
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
    
    def recv_map_and_victims(self, map, victims):
        """ The main rescuer collects all the maps from the 
        explorers and calls the function to unify them. Only the main
        rescuer call this method"""

        print(f"{self.NAME}: Receiving data from explorer...")
        self.collected_maps.append(map)
        self.collected_victims.append(victims)

        if len(self.collected_maps) == self.num_explorers:
            print(f"{self.NAME}: All {self.num_explorers} explorer reported. Commencing unification.")
            self.__unify_and_start_rescue()

    def __unify_and_start_rescue(self):
        """Unify maps, clustering victims and create the other rescuers"""
        for map_obj in self.collected_maps:
            self.unified_map.map_data.update(map_obj.map_data)

        for victims_dict in self.collected_victims:
            self.unified_victims.update(victims_dict)

        self.map = self.unified_map # O agente agora usa o mapa unificado
        self.victims = self.unified_victims

        self.map.draw()

        for seq, data in self.victims.items():
            coord, vital_signals = data
            x, y = coord
            print(f"{self.NAME} Victim {seq} at ({x}, {y}) vs: {vital_signals}")


        # 4.EXECUTAR CLUSTERING E ATRIBUIÇÃO
        self.classifier_victims()
        c1, c2, c3 = self.cluster_victms()
        self._create_and_coordinate_rescuers([c1, c2, c3])
                  
    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        print("\n\n*** R E S C U E R ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        self.map.draw()

        print()
        print(f"{self.NAME} List of found victims received from the explorer")
        self.victims = victims

        # print the found victims - you may comment out
        for seq, data in self.victims.items():
            coord, vital_signals = data
            x, y = coord
            print(f"{self.NAME} Victim {seq} at ({x}, {y}) vs: {vital_signals}")

        #print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.__planner()
        print(f"{self.NAME} PLAN")
        i = 1
        self.plan_x = 0
        self.plan_y = 0
        for a in self.plan:
            self.plan_x += a[0]
            self.plan_y += a[1]
            print(f"{self.NAME} {i}) dxy=({a[0]}, {a[1]}) vic: a[2] => at({self.plan_x}, {self.plan_y})")
            i += 1

        print(f"{self.NAME} END OF PLAN")
                  
        self.set_state(VS.ACTIVE)
        
    def __depth_search(self, actions_res):
        enough_time = True
        ##print(f"\n{self.NAME} actions results: {actions_res}")
        for i, ar in enumerate(actions_res):

            if ar != VS.CLEAR:
                ##print(f"{self.NAME} {i} not clear")
                continue

            # planning the walk
            dx, dy = Rescuer.AC_INCR[i]  # get the increments for the possible action
            target_xy = (self.plan_x + dx, self.plan_y + dy)

            # checks if the explorer has not visited the target position
            if not self.map.in_map(target_xy):
                ##print(f"{self.NAME} target position not explored: {target_xy}")
                continue

            # checks if the target position is already planned to be visited 
            if (target_xy in self.plan_visited):
                ##print(f"{self.NAME} target position already visited: {target_xy}")
                continue

            # Now, the rescuer can plan to walk to the target position
            self.plan_x += dx
            self.plan_y += dy
            difficulty, vic_seq, next_actions_res = self.map.get((self.plan_x, self.plan_y))
            #print(f"{self.NAME}: planning to go to ({self.plan_x}, {self.plan_y})")

            if dx == 0 or dy == 0:
                step_cost = self.COST_LINE * difficulty
            else:
                step_cost = self.COST_DIAG * difficulty

            #print(f"{self.NAME}: difficulty {difficulty}, step cost {step_cost}")
            #print(f"{self.NAME}: accumulated walk time {self.plan_walk_time}, rtime {self.plan_rtime}")

            # check if there is enough remaining time to walk back to the base
            if self.plan_walk_time + step_cost > self.plan_rtime:
                enough_time = False
                #print(f"{self.NAME}: no enough time to go to ({self.plan_x}, {self.plan_y})")
            
            if enough_time:
                # the rescuer has time to go to the next position: update walk time and remaining time
                self.plan_walk_time += step_cost
                self.plan_rtime -= step_cost
                self.plan_visited.add((self.plan_x, self.plan_y))

                if vic_seq == VS.NO_VICTIM:
                    self.plan.append((dx, dy, False)) # walk only
                    #print(f"{self.NAME}: added to the plan, walk to ({self.plan_x}, {self.plan_y}, False)")

                if vic_seq != VS.NO_VICTIM:
                    # checks if there is enough remaining time to rescue the victim and come back to the base
                    if self.plan_rtime - self.COST_FIRST_AID < self.plan_walk_time:
                        print(f"{self.NAME}: no enough time to rescue the victim")
                        enough_time = False
                    else:
                        self.plan.append((dx, dy, True))
                        #print(f"{self.NAME}:added to the plan, walk to and rescue victim({self.plan_x}, {self.plan_y}, True)")
                        self.plan_rtime -= self.COST_FIRST_AID

            # let's see what the agent can do in the next position
            if enough_time:
                self.__depth_search(self.map.get((self.plan_x, self.plan_y))[2]) # actions results
            else:
                return

        return
    
    def __planner(self):
        """
        Planeja a sequência de resgates baseada na lista de vítimas atribuída (self.victims).
        """
        curr_x, curr_y = 0, 0 # Base (assumindo que o agente começa na base relativa 0,0 do seu sistema)
        
        current_rtime = self.TLIM
        
        # Lista para salvar as vítimas salvas efetivamente para o arquivo de saída
        self.victims_saved_sequence = [] 

        victims_list = []
        for seq, data in self.victims.items():
            pos = data[0]
            victims_list.append({'id': seq, 'pos': pos})
        
        for vic in victims_list:
            target_pos = vic['pos']
            
            path_to_vict, cost_to_vict = self.a_star((curr_x, curr_y), target_pos, self.map.map_data)
            
            if not path_to_vict:
                continue
                
            path_to_base, cost_to_base = self.a_star(target_pos, (0,0), self.map.map_data)
            
            if not path_to_base:
                continue

            total_cost = cost_to_vict + self.COST_FIRST_AID + cost_to_base
            
            if total_cost <= current_rtime:
                for move in path_to_vict:
                    self.plan.append((move[0], move[1], False))

                last_move = self.plan.pop()
                self.plan.append((last_move[0], last_move[1], True))
                
                current_rtime -= (cost_to_vict + self.COST_FIRST_AID)
                curr_x, curr_y = target_pos
                
                self.victims_saved_sequence.append(vic)

        path_home, _ = self.a_star((curr_x, curr_y), (0,0), self.map.map_data)
        if path_home:
            for move in path_home:
                self.plan.append((move[0], move[1], False))
                
           
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           #input(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if there_is_vict:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

    def classifier_victims(self):
        """ Classify the found victims using the pre-trained classifier
            and print the results.
        """
        for seq, ((position),victm) in self.victims.items():
            # Prepare the feature vector for classification
            features = victm[1:11]
            colunas = [
                "idade","fc","fr","pas","spo2","temp","pr","sg","fx","queim"
            ]
            df = pd.DataFrame([features], columns=colunas)
            # Predict the class using the loaded classifier
            prediction = self.classifier.predict(df)
            distance  = sqrt(position[0]**2 + position[1]**2)
            self.cluster_data.append([distance, prediction[0], position, seq])

    def cluster_victms(self):
        """ Cluster the victims based on distance and predicted severity using KMeans.
            Print the cluster centers and labels.
        """
        if not self.cluster_data:
            print("No victim data to cluster.")
            return
        cluster_features = [[data[0], data[1]] for data in self.cluster_data]
        print(f"cluster features: {cluster_features}")
        kmeans = KMeans(n_clusters=3, random_state=0)
        labels = kmeans.fit(cluster_features).labels_

        x = [f[0] for f in cluster_features]  # distância
        y = [f[1] for f in cluster_features]  # tri

        # Criar figura
        plt.figure(figsize=(8, 6))

        # Plotar pontos (cada cor = cluster)
        plt.scatter(x, y, c=labels, s=60, alpha=0.8, edgecolors='black')

        # Personalização dos eixos
        plt.title("Clusters de Vítimas (Distância vs Tri)")
        plt.xlabel("Distância")
        plt.ylabel("Tri (gravidade 0 a 3)")
        plt.yticks([0, 1, 2, 3])
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        # Mostrar o gráfico
        plt.show()
        
        cluster_df = pd.DataFrame([
            {
                "id_vict": data[3],         # seq (identificador)
                "x": data[2][0],            # posição x
                "y": data[2][1],            # posição y
                "tri": int(data[1]),        # classe predita (gravidade)
                "cluster": int(label)
            }
            for data, label in zip(self.cluster_data, labels)
        ])

        # Salva cada cluster separadamente
        for c in sorted(cluster_df["cluster"].unique()):
            df_cluster = cluster_df[cluster_df["cluster"] == c][["id_vict", "x", "y", "tri"]]
            nome_arquivo = f"cluster{c+1}.txt"
            df_cluster.to_csv(nome_arquivo, index=False)
        
        cluster_1 = cluster_df[cluster_df["cluster"] == 0]
        cluster_2 = cluster_df[cluster_df["cluster"] == 1]
        cluster_3 = cluster_df[cluster_df["cluster"] == 2]

        return cluster_1, cluster_2, cluster_3

    def _create_and_coordinate_rescuers(self, clusters):
        """Creates and coordinates other rescuers, distributing clusters among them."""
        # Create 2 additional rescuers
        rescuers = [self]
        #create other rescuers
        for i in range(1,3):
            curr = os.getcwd()
            config_ag_folder = os.path.join(curr, "config_ag_1")
            rescuer_file = os.path.join(config_ag_folder, f"rescuer_{i+1}.txt")

            new_rescuer = Rescuer(self.get_env(), rescuer_file)
            new_rescuer.map = self.map
            rescuers.append(new_rescuer)
            print(f"{self.NAME}: Created rescuer {new_rescuer.NAME}")

        # Distribute clusters to other rescuers
        victms_copy = self.victims.copy()
        for i, rescuer in enumerate(rescuers):
            if i < len(clusters):  # +1 because master takes first cluster
                cluster = clusters[i]
                # Share necessary information with the rescuer
                rescuer.victims = {
                    id_victm: victms_copy[id_victm]
                    for id_victm in cluster["id_vict"].tolist()
                    if id_victm in victms_copy
                }
                print(f"{rescuer.NAME}: Assigned cluster {i} to {rescuer.NAME}")


        for rescuer in rescuers:
            
            print(f"Vitimas do rescuer: {self.victims}")
            rescuer.__planner()
            rescuer.set_state(VS.ACTIVE)

            print(f"{rescuer.NAME} PLAN")
            for victm in rescuer.victims.items():
                print(f"{rescuer.NAME} Victim to rescue: {victm}")

            print(f"{rescuer.NAME} END OF PLAN")

    def order_victims(self):

        #parameters to genetic algorithm
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 1500
        TOURNAMENT_SIZE = 50
        MUTATION_RATE = 0.1
        CROSSOVER_RATE = 0.8

        best_sequence = []
        best_fitness = 0

        victims = self.victims

        def selection(population, fitness):
            tournament = random.sample(list(zip(population, fitness)), TOURNAMENT_SIZE)
            winner = max(tournament, key=lambda item: item[1])
            return winner[0]

        def crossover (parent1, parent2):
            if(random.random > CROSSOVER_RATE):
                    return parent1, parent2
            size = len(parent1)
            child1, child2 = [-1]*size, [-1]*size
            

            #init and final os subsequence of parent1
            start, end = sorted(random.sample(range(size), 2))

            #mix parent gens
            child1[start:end+1] = parent1[start:end+1]
            child2[start:end+1] = parent2[start:end+1]

            for i in range(size):
                if child1[i] == -1:
                    child1[i] = parent2[i]
                if child2[i] == -1:
                    child2[i] = parent1[i]
            return child1, child2
        
        def mutation(ind):
            #swap 2 gens
            if random.random < MUTATION_RATE:
                gen1, gen2 = random.sample(range(len(ind)), 2)
                ind[gen1], ind[gen2] = ind[gen2], ind[gen1]
            return ind
        
        def calc_fitness():
            pass


        #init population
        population = []
        for _ in range(POPULATION_SIZE):
            new_sequence = random.sample(victims, len(victims))
            population.append(new_sequence)

        #evolution
        fit = []
        for gen in NUM_GENERATIONS:
            for ind in population:
                fit.append(calc_fitness(ind))
            best_fitness = max(fit)
            best_sequence = population[fit.index(best_fitness)]

            #next generation
            new_pop = []

            while len(new_pop) < POPULATION_SIZE:
                #select parents
                parent1 = selection(population, fit)
                parent2 = selection(population, fit)

                #2 childrens crossover and mutation
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent1, parent2)

                child1 = mutation(child1)
                child2 = mutation(child2)

                new_pop.append(child2, child2)

            population = new_pop
        self.victims = best_sequence

            
                






    
    def a_star(self, start, goal, map):
        """ returns the tuple [path, cost]"""
        
        open_set = []
        #Heap: (f_score, g_score, pos, path)
        heapq.heappush(open_set, (0, 0, start, []))

        visited = set()
        g_scores = {start: 0}

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current == goal:
                return path, g_score

            if current in visited:
                continue

            visited.add(current)

            for action_idx in range (8):
                dx, dy = Rescuer.AC_INCR[action_idx]
                neighbour = (current[0] + dx, current[1] + dy)

                data = map.get(neighbour)

                if not data: continue
                diff = data[0]
                if diff == VS.OBST_WALL: continue

                move_cost = self.COST_LINE if (dx == 0 or dy == 0) else self.COST_DIAG
                g1_score = g_score + move_cost * diff

                if neighbour not in g_scores or g1_score < g_scores[neighbour]:
                    g_scores[neighbour] = g1_score
                    
                    h = self.heuristic(neighbour, goal)
                    f = g1_score + h
                    new_path = path + [(dx,dy)]
                    heapq.heappush(open_set, (f, g1_score, neighbour, new_path))
        return None, float('inf')
    
    def heuristic(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5 * self.COST_LINE
