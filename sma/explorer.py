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
    def __init__(self, env, config_file, resc, target_pos=(0,0)):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.target_pos = target_pos

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
        self.a_star_path = []     
        #add possible actions from the starting position to untried
        self.untried[(self.x,self.y)] = self.get_possible_actions()

        self.obst_min = env.get_obst_min()

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        
    def get_possible_actions(self):
        obstacles_start = self.check_walls_and_lim()
        possible_actions = []
        for i, s in enumerate(obstacles_start):
            if s == VS.CLEAR:
                dx, dy = Explorer.AC_INCR[i]
                target_pos = (self.x + dx, self.y + dy)
                
                if not self.map.in_map(target_pos):
                    dist_sq = (target_pos[0] - self.target_pos[0])**2 + (target_pos[1] - self.target_pos[1])**2
                    possible_actions.append((i, dist_sq))

        possible_actions.sort(key=lambda x: x[1], reverse=True)

        actions = [item[0] for item in possible_actions]
        
        return actions

    def explore(self):
        # Online-DFS
        current_state = (self.x, self.y)

        if current_state not in self.untried:
            self.untried[current_state] = self.get_possible_actions()

        while self.untried.get(current_state):
            action_idx = self.untried[current_state].pop()
            
            dx, dy = Explorer.AC_INCR[action_idx]
            target_pos = (self.x + dx, self.y + dy)

            if self.map.in_map(target_pos):
                continue
            
            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()

            if result == VS.BUMPED:
                self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
                continue 

            if result == VS.EXECUTED:
                self.walk_stack.push((dx, dy))
                self.x += dx
                self.y += dy          

                next_state = (self.x, self.y)
                
                if next_state not in self.untried:
                    self.untried[next_state] = self.get_possible_actions()

                    seq = self.check_for_victim()
                    if seq != VS.NO_VICTIM:
                        vs = self.read_vital_signals()
                        self.victims[seq] = ((self.x, self.y), vs)
                
                    difficulty = (rtime_bef - rtime_aft)
                    if dx == 0 or dy == 0:
                        difficulty = difficulty / self.COST_LINE
                    else:
                        difficulty = difficulty / self.COST_DIAG

                    self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

                return

        if self.walk_stack.is_empty():
            self.returning_to_base = True
            return

        self.come_back()
        return

    def come_back(self):
        """ Procedure to return to the base using A* calculated path """
        
        if self.returning_to_base and self.a_star_path:
            next_pos = self.a_star_path.pop(0)
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            
            result = self.walk(dx, dy)

            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy
            return

        if not self.walk_stack.is_empty():
            dx, dy = self.walk_stack.pop()
            
            back_dx, back_dy = -dx, -dy
            
            result = self.walk(back_dx, back_dy)
            if result == VS.EXECUTED:
                self.x += back_dx
                self.y += back_dy
            elif result == VS.BUMPED:
                print(f"{self.NAME}: Unexpected bump while coming back at ({self.x + back_dx}, {self.y + back_dy})")


        
    def deliberate(self) -> bool:
        """  The simulator calls this method at each cycle. 
        Must be implemented in every agent. The agent chooses the next action.
        """

        # Only calculate A* path cost when still exploring
        if not self.returning_to_base:
            path_info = self.a_star((self.x, self.y), (0, 0), self.map.map_data)
            
            if path_info:
                path, cost = path_info
                safety_margin = cost * 1.16
                
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
                self.resc.recv_map_and_victims(self.map, self.victims)
                return False
            self.come_back()
            return True
        
        self.explore()
        return True
    
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
                dx, dy = Explorer.AC_INCR[action_idx]
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
                    new_path = path + [neighbour]
                    heapq.heappush(open_set, (f, g1_score, neighbour, new_path))
        return None, float('inf')
    
    def heuristic(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5 * self.obst_min
