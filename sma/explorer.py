# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.


import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from joblib import load
import pandas as pd
from math import sqrt
from sklearn.cluster import KMeans

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, action_order):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is a seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.classifier = load('classifier.joblib')  # load the pre-trained classifier
        self.cluster_data = []
        self.untried = {}          # dictionary of untried actions: (x,y) -> [actions_id] 
        self.returning_to_base = False
        self.action_order = action_order
        #add possible actions from the starting position to untried
        self.untried[(self.x,self.y)] = self.get_possible_actions()

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        
    def get_possible_actions(self):
        obstacles_start = self.check_walls_and_lim()
        possible_actions = []
        for i, s in enumerate(obstacles_start):
            if s == VS.CLEAR:
                possible_actions.append(i)

        sorted_possible_actions = []
        for preferred_action in self.action_order:
            if preferred_action in possible_actions:
                sorted_possible_actions.append(preferred_action)
        sorted_possible_actions.reverse()
        return sorted_possible_actions

    def explore(self):
        # Online-DFS

        current_sate = (self.x, self.y)

        if not self.untried.get(current_sate):
            #beco sem saida
            if self.walk_stack.is_empty():
                #chegou na base
                self.returning_to_base = True
                return
            self.come_back()
            return

        action_index = self.untried.get(current_sate).pop()
        dx, dy = Explorer.AC_INCR[action_index]

        # Moves the explorer agent to another position
        rtime_bef = self.get_rtime()   ## get remaining batt time before the move
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()   ## get remaining batt time after the move

        # Test the result of the walk action
        # It should never bump, since get_next_position always returns a valid position...
        # but for safety, let's test it anyway
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # puts the visited position in a stack. When the batt is low, 
            # the explorer unstack each visited position to come back to the base
            self.walk_stack.push((dx, dy))
            # update the agent's position relative to the origin of 
            # the coordinate system used by the agents
            self.x += dx
            self.y += dy          

            next_state = (self.x, self.y)

            if next_state not in self.untried:
                self.untried[next_state] = self.get_possible_actions()

                # Check for victims
                seq = self.check_for_victim()
                if seq != VS.NO_VICTIM:
                    vs = self.read_vital_signals()
                    self.victims[seq] = ((self.x, self.y), vs)
                    #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                    #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
                # Calculates the difficulty of the visited cell
                difficulty = (rtime_bef - rtime_aft)
                if dx == 0 or dy == 0:
                    difficulty = difficulty / self.COST_LINE
                else:
                    difficulty = difficulty / self.COST_DIAG

                # Update the map with the new cell
                self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
                #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        """ Procedure to return to the base: pops the walk_stack to follow
        the exploration path in the opposite direction """
  
        if self.walk_stack.is_empty():
            return

        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        # Walk resulted in bumping into a wall or end of grid
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
            
        # Walk succeded
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")


        
    def deliberate(self) -> bool:
        """  The simulator calls this method at each cycle. 
        Must be implemented in every agent. The agent chooses the next action.
        """

        consumed_time = self.TLIM - self.get_rtime()
        # check if it is time to come back to the base      
        # Verifica se é hora de voltar (sua estratégia de "metade do tempo")
        if not self.returning_to_base and (consumed_time >= self.get_rtime()):
            self.returning_to_base = True
            print(f"{self.NAME}: Returning to base.")

        if self.returning_to_base:
            if (self.x == 0 and self.y == 0):
                self.classifier_victims()
                self.cluster_victms()
                # time to wake up the rescuer
                # pass the walls and the victims (here, they're empty)
                print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the orchestrator rescuer")
                #input(f"{self.NAME}: type [ENTER] to proceed")
                self.resc.recv_map_and_victims(self.map, self.victims)
                return False
            self.come_back()
            return True
        
        self.explore()
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

