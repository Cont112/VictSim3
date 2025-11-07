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
import heapq

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
        self.untried = {}          # dictionary of untried actions: (x,y) -> [actions_id] 
        self.returning_to_base = False
        self.action_order = action_order
        self.a_star_path = []     
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
        """ Procedure to return to the base using A* calculated path """
        
        if not self.a_star_path:
            return

        next_pos = self.a_star_path.pop(0)
        dx = next_pos[0] - self.x
        dy = next_pos[1] - self.y

        result = self.walk(dx, dy)
        # Walk resulted in bumping into a wall or end of grid
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
            
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")


        
    def deliberate(self) -> bool:
        """  The simulator calls this method at each cycle. 
        Must be implemented in every agent. The agent chooses the next action.
        """

        # Only calculate A* path cost when still exploring
        if not self.returning_to_base:
            path_info = self.a_star()
            
            if path_info:
                path, cost = path_info
                safety_margin = cost * 1.08
                
                if self.get_rtime() <= safety_margin:
                    self.returning_to_base = True
                    self.a_star_path = path
                    print(f"{self.NAME}: Returning to base. A* cost: {cost:.2f}, with margin: {safety_margin:.2f}, rtime: {self.get_rtime():.2f}")
            else:
                consumed_time = self.TLIM - self.get_rtime()
                if consumed_time >= self.get_rtime():
                    self.returning_to_base = True
                    print(f"{self.NAME}: Returning to base (no A* path found).")

        if self.returning_to_base:
            if (self.x == 0 and self.y == 0):
                print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the orchestrator rescuer")
                #input(f"{self.NAME}: type [ENTER] to proceed")
                self.resc.recv_map_and_victims(self.map, self.victims)
                return False
            self.come_back()
            return True
        
        self.explore()
        return True
    
    def a_star(self):
        """ A* pathfinding to return to base
        @returns: tuple (path, cost) where:
                  - path is a list of (x,y) coordinates from current position to base (excluding current pos)
                  - cost is the total battery cost to traverse the path
                  Returns None if no path is found
        """
        start = (self.x, self.y)
        goal = (0, 0)
        
        if start == goal:
            return ([], 0)
        
        counter = 0
        heap = [(0, counter, start, [], 0)]
        counter += 1
        
        best_g_score = {start: 0}
        
        while heap:
            f_score, _, current, path, g_score = heapq.heappop(heap)
            
            if current == goal:
                return (path, g_score)
            
            if current in best_g_score and g_score > best_g_score[current]:
                continue
            
            for action_idx in range(8):
                dx, dy = Explorer.AC_INCR[action_idx]
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.map.in_map(neighbor):
                    continue
                
                neighbor_data = self.map.get(neighbor)
                difficulty = neighbor_data[0]
                
                if difficulty == VS.OBST_WALL:
                    continue
                
                if dx == 0 or dy == 0:
                    move_cost = self.COST_LINE * difficulty
                else:
                    move_cost = self.COST_DIAG * difficulty
                
                new_g_score = g_score + move_cost
                
                if neighbor not in best_g_score or new_g_score < best_g_score[neighbor]:
                    best_g_score[neighbor] = new_g_score
                    
                    h_score = (abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])) * self.COST_LINE
                    
                    new_f_score = new_g_score + h_score
                    
                    new_path = path + [neighbor]
                    heapq.heappush(heap, (new_f_score, counter, neighbor, new_path, new_g_score))
                    counter += 1
        return None