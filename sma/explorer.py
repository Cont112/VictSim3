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
    def __init__(self, env, config_file, resc):
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

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
    
        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            if obstacles[direction] == VS.CLEAR:
                return Explorer.AC_INCR[direction]
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

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

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[seq] = ((self.x, self.y), vs)
                print(f"Vitima encontrada. Sinais: {vs}\n")
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
        if consumed_time < self.get_rtime():
            # continue to explore
            self.explore()
            return True

        # Returning to the base terminates when there are no more moves to pop from the stack
        # or when the agent reaches (0, 0) — the base position
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            self.classifier_victims()
            self.cluster_victms()
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            #input(f"{self.NAME}: type [ENTER] to proceed")
            self.resc.go_save_victims(self.map, self.victims)
            return False

        # move to the base
        self.come_back()
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

