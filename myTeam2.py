# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from game import Actions
from game import Directions
import time
import math as m
import numpy as np
from util import nearestPoint
# import itertools
# import pdb
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ProtectingAgent', second = 'AttackingAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
      
  recent_moves = []
  possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
      
  def registerInitialState(self, gameState):
    """
    registering initial state for the base class
    """
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.possible_directions = set(self.possible_directions)

  def isGoalState(self, myPos, goal_pos):
    """
    to test goal state of astar path
    """
    return myPos == goal_pos

  def heuristic(self, mypos, startpos):
    """
    the heuristic for astar path
    guidance by the distance from the from location
    """
    return self.getMazeDistance(mypos, startpos)

  def aStarSearch(self, gameState, myPos, goal_pos):
    """
    astar search to find paths between two given position
    """
    myPQ = util.PriorityQueue()
    startPos = myPos
    startNode = (startPos, '',0, [])
    myPQ.push(startNode,self.heuristic(startPos,startPos))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
      node = myPQ.pop()
      curPos, action, cost, path = node
      if (not curPos in visited) or cost < best_g.get(curPos):
        visited.add(curPos)
        best_g[curPos]=cost
        if self.isGoalState(curPos, goal_pos):
          path = path + [(curPos, action)]
          actions = [action[1] for action in path]
          del actions[0]
          return actions
        all_actions = self.possible_directions

        for succAction in all_actions:
          succ_x, succ_y = Actions.getSuccessor(curPos, succAction)
          succPos = (int(succ_x), int(succ_y))
          if (gameState.hasWall(int(succ_x), int(succ_y))): continue
          newNode = (succPos, succAction, cost + 1, path + [(node, action)])
          myPQ.push(newNode,self.heuristic(succPos,startPos)+cost+1)

  def findCenterPos(self, gameState):
    """
    takes gamestate as input to find the center location
    used to find all the points from where the attacking agents can start attack
    """
    total_width = gameState.data.layout.width
    total_height = gameState.data.layout.height
    offset = 0
    detour_offset = -1
    if gameState.isOnRedTeam(self.index):
      offset = -1
      detour_offset = 0
    
    if not self.home_x:
      self.home_x = m.floor(total_width / 2)

    for y in range(total_height):
      if not gameState.hasWall(self.home_x + offset, y) :
        self.escape_pos.append((self.home_x + offset, y))

    return

  def evaluate(self, state, action, depth):
    """
    rewards and penalty are shaped in terms of features and weights
    The features and weights vectors are multiplied here to return rewards
    """
    features = self.getFeatures(state, action, depth)
    weights = self.getWeights(state, action)
    return features * weights
  
  def isStuck(self):
    """
    evaluates if the agent is stuck in a location
    """
    if len(self.recent_moves) > 4:
      self.recent_moves = self.recent_moves[-4:]
    if len(self.recent_moves) < 4:
      return False

    if ((len(set(self.recent_moves)) == len(self.possible_directions)) or 
        (self.recent_moves[0] != self.recent_moves[1] and 
        self.recent_moves[0] == self.recent_moves[2] and self.recent_moves[1] == self.recent_moves[3])):
      self.stuck_counter += 1
      if (self.stuck_counter > 2):
        return True
    else:
      self.stuck_counter = 0
    return False

class AttackingAgent(ReflexCaptureAgent):

  max_distance = None
  home_x = None
  capsules = []
  food = None
  start_pos = None
  aStarPath = []
  pathToAttackPos = []
  attack_path_counter = 0
  deepest_trapped_food = None
  escape_pos = []
  food_count = None
  is_going_home = False
  stuck_counter = 0
  recent_moves = []
  trapped_food = util.PriorityQueue()
  inactivity = 0
  atHome_count = 0
  trap_list = []
  trap_depths = util.Counter()
  trapped_food_paths = util.Counter()
  inTrap = False
  ghost_seen = -1
  max_trap_depth = 0
  max_eval_value = 99999
  entry_point = None
  max_distance = 40
  numCarrying = 0

  #position for foods that is not in trap
  def getUntrappedFood(self, gameState, food_list) :
    not_trapped_list = []
    for food in food_list:
      legalNeighbours = Actions.getLegalNeighbors(food, gameState.getWalls())
      legalNeighbours.remove(food)
      if (len(legalNeighbours) > 2) :
        not_trapped_list.append(food)
    return not_trapped_list

  def setDeepTrappedFood(self, gameState):
    """
    list of the deepest trapped food that pacman can access 
    and list of trapped locations are created
    """
    total_width = gameState.data.layout.width
    total_height = gameState.data.layout.height
    neighboursQ = []
    self.trapped_food = util.PriorityQueue()
    for food in self.food:
      legalNeighbours = Actions.getLegalNeighbors(food, gameState.getWalls())
      legalNeighbours.remove(food)
      if len(legalNeighbours) > 1:
        continue
      neighboursQ = []
      visitedQ = [food]
      if food not in self.trap_list:
        self.trap_list.append(food)
        self.trap_depths[food] = 0
      for nei in legalNeighbours:
        neighboursQ.append(nei)
        steps = 0
      while len(neighboursQ) > 0:
        next_nei = neighboursQ.pop(0)
        visitedQ.append(next_nei)
        if(next_nei in legalNeighbours):
          steps = 0
        steps += 1
        neighbours = Actions.getLegalNeighbors(next_nei, gameState.getWalls())
        neighbours.remove(next_nei)

        if len(neighbours) > 2:
          self.trapped_food.push(food, (steps * -1))
          for capsule in self.capsules:
            self.trapped_food_paths[capsule+food] = self.aStarSearch(gameState, capsule, food)
          if steps > self.max_trap_depth:
            self.max_trap_depth = steps
          continue

        if next_nei not in self.trap_list:
          self.trap_list.append(next_nei)
          self.trap_depths[next_nei] = steps

        for nei in neighbours:
          if nei not in visitedQ and nei not in neighboursQ:
            neighboursQ.append(nei)

  #This method handles the initial setup of the agent
  def registerInitialState(self, gameState):
    self.start_pos = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.capsules = self.getCapsules(gameState)
    self.food = self.getFood(gameState).asList()
    self.findCenterPos(gameState)
    for pos in self.escape_pos:
      self.pathToAttackPos.append(self.aStarSearch(gameState, self.start_pos, pos))
      self.attack_path_counter += 1
    self.setDeepTrappedFood(gameState)
    self.food_count = len(self.food)

  def findInRangeGhost(self, state, myPos) :
    """
    Compute distance to the nearest ghost
    and find if there are ghosts in range
    """
    inRange = []
    closestDist = float('inf')
    enemy_indices = self.getOpponents(state)
    for index in enemy_indices:
      enemy_agent = state.getAgentState(index)
      enemy_pos = enemy_agent.getPosition()
      if (enemy_pos and self.getMazeDistance(myPos, enemy_pos) < 10 and 
          ((self.getMazeDistance(myPos, enemy_pos) < 7 and 
          enemy_agent.scaredTimer < 7 and not enemy_agent.isPacman))):
        inRange.append(enemy_agent)
        if self.getMazeDistance(myPos, enemy_pos) < closestDist:
          closestDist = self.getMazeDistance(myPos, enemy_pos) 
    return inRange, closestDist

  def getFeatures(self, state, action, depth):
    features = util.Counter()
    pos = state.getAgentState(self.index).getPosition()
    successor = state.generateSuccessor(self.index, action)
    myPos = successor.getAgentState(self.index).getPosition()

    # Compute score from successor state
    features['successorScore'] = self.getScore(successor)    
    # in general give preference to being a pacman
    features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

    inRange, closestDist = self.findInRangeGhost(successor, myPos)
    closestDist += (3 - depth)
    distanceToHome = min([self.getMazeDistance(myPos, pos) for pos in self.escape_pos])

    if self.entry_point:
      """
      if entry_point has value, it would indicate pacman requires diversion to go out.
      pacman is encouraged to go out.
      """
      features['isPacman'] = 0 if successor.getAgentState(self.index).isPacman else 1
      features['distanceToEntry'] = self.getMazeDistance(myPos, self.entry_point)
      # self.getMazeDistance(myPos, self.entry_point)
      return features

    # when no ghost seen
    features['distanceToGhost'] = self.max_distance

    # Compute distance to the nearest food
    foodList = self.food

    features['location_trapped'] = -1
    features['trap_depth'] = self.max_trap_depth + 2

    # Compute distance to nearest ghost
    if (len(inRange) > 0 or self.ghost_seen > -1):
          
      features['distanceToGhost'] = closestDist
      
      # checks if location is trapped   
      features['location_trapped'] = -1
      # also checks the locations trap depth
      features['trap_depth'] = self.max_trap_depth + 2
      if myPos in self.trap_list:
        features['location_trapped'] = 1
        features['trap_depth'] = self.trap_depths[myPos]
      
      # if pacman is chased by ghost get untrapped food
      if (successor.getAgentState(self.index).isPacman and 
          (closestDist < 2 or distanceToHome > 4)) :
        foodList = self.getUntrappedFood(successor, foodList)
        if (len(foodList) == 0 and 
          successor.getAgentState(self.index).numCarrying == 0):
          foodList = self.food

      #returns position of the nearest capsule  
      if len(self.capsules) > 0:
        minDistance = 0
        if (myPos not in self.capsules):
          minDistance = min([self.getMazeDistance(myPos, food) for food in self.capsules])
        features['distanceToCapsule'] = minDistance
    
    elif (self.numCarrying < (self.food_count * (2 / 3)) or
      state.data.timeleft > 40):
      self.is_going_home = False

    # returns distance to nearest Food
    if (len(foodList) > 0):
      minDistance = 0
      if (myPos not in self.food):
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance


    # compute minimum distane to home
    if self.is_going_home:
      features['distanceToHome'] = distanceToHome
      features['isPacman'] = 0 if successor.getAgentState(self.index).isPacman else 1

    return features

  def getWeights(self, state, action):
    """
    Get weights for the features used in the evaluation.
    """
    weights = util.Counter()

    successor = state.generateSuccessor(self.index, action)
    myPos = successor.getAgentState(self.index).getPosition()

    weights['isPacman'] = 2
    weights['location_trapped'] = -5
    weights['trap_depth'] = 5
    weights['distanceToGhost'] = 6
    weights['successorScore'] = 2
    weights['distanceToFood'] = -5

    if self.entry_point:
      weights['distanceToEntry'] = -10
      weights['isPacman'] = 15

    distanceToHome = min([self.getMazeDistance(myPos, pos) for pos in self.escape_pos])
    inRange, closestDist = self.findInRangeGhost(successor, myPos)
    if len(inRange) > 0 or self.ghost_seen > -1:
      weights['distanceToFood'] = -1
      weights['distanceToCapsule'] = -6
      weights['location_trapped'] = -5
      # weights['trap_depth'] = 5
      print(self.numCarrying)
      if (self.numCarrying < 4):
        # (self.numCarrying == 0 and distanceToHome < 5) or 
        weights['distanceToGhost'] = 4
        weights['distanceToCapsule'] = -3
        weights['distanceToFood'] = -5
        weights['isPacman'] = 5
      if (not successor.getAgentState(self.index).isPacman):
        weights['distanceToGhost'] = 1
      if myPos in self.trap_list:
        weights['distanceToGhost'] = -8

    if self.is_going_home:
      weights['distanceToHome'] = -2
      weights['distanceToFood'] = -1

    return weights

  def simulate(self, state, depth, path, parent_value, isGhostInRange):
    """
    Simulates positions at a given depth
    """
    mypos = state.getAgentPosition(self.index)
    path.append(mypos)
    search_list = self.getFood(state).asList()
    ghost_list = []
    inRange, closestDist = self.findInRangeGhost(state, mypos)
    if (len(inRange)):
      ghost_list = [enemy_agent.getPosition() for enemy_agent in inRange]
      search_list = ghost_list
      self.ghost_seen = closestDist
      

    self_eval = self.evaluate(state, Directions.STOP, depth)
    # isGhostInRange says if parent could see ghost
    if isGhostInRange and len(inRange) == 0 and state.getAgentState(self.index).isPacman:
      # if parent could see and current location is capsule, then capsule is important
      if mypos in self.capsules:
        return (self.max_eval_value, 1)
      # else return value of parent which could see ghost
      if parent_value:
        return parent_value, 1
    

    if(parent_value!=None and self_eval < parent_value):
      # if parent's value is not none and current value has dropped from parent's value
      if not isGhostInRange and len(inRange) > 0:
        # return own value if ghost can be seen here
        return self_eval, 1
      else:
        # return parent's higher value otherwise, assuming the drop is 
        # due to reasons like going away from food
        return parent_value, 1

    if not isGhostInRange and len(inRange) > 0:
      isGhostInRange = True

    if mypos in search_list or depth == 0:
      # if current position is in either food list or ghost list or the depth of search is over
      # return own value
      return self_eval, 1 
  
    # if(mypos in search_list or depth == 0 
    #   or (parent_value!=None and self_eval < parent_value)):
    #   return (self_eval, 1)

    # simulate child positions
    legal_actions = state.getLegalActions(self.index)
    legal_actions.remove(Directions.STOP)

    ## Preventing agent from taking reverse direction
    reversed_direction = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
    if (reversed_direction in legal_actions and
        len(legal_actions) > 1 and
        state.getAgentState(self.index).isPacman):
      legal_actions.remove(reversed_direction)

    n = 1
    val = []
    to_rem = []
    for action in legal_actions:
      state_prime = state.generateSuccessor(self.index, action)
      pos_prime = state_prime.getAgentPosition(self.index)
      if pos_prime in self.capsules and isGhostInRange:
        v = self.max_eval_value
        continue
      if((pos_prime in path) or (pos_prime == self.start_pos)):
        # remove loops in path
        # remove any possibility to be eaten by ghost
        to_rem.append(action)
        continue
      v, new_n = self.simulate(state_prime, depth - 1, path, self_eval, isGhostInRange)
      val.append(v)
      if (new_n + 1) > n: n = new_n + 1

    for rem_action in to_rem:
      legal_actions.remove(rem_action)

    if(len(val) == 0):
      return (self_eval, n - 1)

    return (max(val), n)

  #the agent try to find new entry point when start next attack
  def findNewEntryPoint(self, gameState, myPos):
    my_x, my_y = myPos
    maxDist = 0
    for pos in self.escape_pos:
      dist = self.getMazeDistance(myPos,pos)
      if dist > maxDist:
        maxDist = dist
        self.entry_point = pos
  

 #find minimum distance to home
  def findDistanceToHome(self, myPos):
    minDistance = 9999
    for position in self.escape_pos:
      if self.getMazeDistance(myPos, position) < minDistance:
        minDistance = self.getMazeDistance(myPos, position)

    return minDistance

  def chooseAction(self, gameState):
    agent = gameState.getAgentState(self.index)
    agentLocation = gameState.getAgentPosition(self.index)
    print(gameState)
   # if pacman on his own side, so reset the num carrying timer
    self.inTrap = False
    self.ghost_seen = -1
    self.numCarrying = agent.numCarrying

    # pacman is starting from the start position
    if agentLocation == self.start_pos:
      path_index = random.randint(0, self.attack_path_counter - 1)
      self.aStarPath = self.pathToAttackPos[path_index]
      self.aStarPath = self.aStarPath[:-2]

    # after pacman eats the deepest trapped food of the moment
    if agentLocation in self.trap_list:
      self.inTrap = True


    if agentLocation == self.deepest_trapped_food:
      self.deepest_trapped_food = None

      # print("I'm going home")

    if not agent.isPacman:
      self.food = self.getFood(gameState).asList()
      self.is_going_home = False
      self.food_count = len(self.food)
      self.inactivity = 0


    if (agentLocation in self.capsules) :
      self.capsules.remove(agentLocation)
      while (not self.trapped_food.isEmpty() and not self.deepest_trapped_food):
        self.deepest_trapped_food = self.trapped_food.pop()
        self.aStarPath = self.trapped_food_paths[agentLocation+self.deepest_trapped_food]
        if self.deepest_trapped_food not in self.food or self.aStarPath == 0:
          self.deepest_trapped_food = None
          self.aStarPath = []

    if agentLocation in self.food:
      self.food.remove(agentLocation)

    isGhostInRange = False
    inRange, closestDist = self.findInRangeGhost(gameState, agentLocation)

    if ((agent.numCarrying > (self.food_count * (1 / 3)) and 
        len(inRange) > 0 and closestDist < 8) or 
        (agent.numCarrying > (self.food_count * (9 / 10))) or
        (agent.numCarrying > 0 and gameState.data.timeleft < 40)):
      self.is_going_home = True

     # the agent will select new path to follow when start new attack
    
    if len(self.aStarPath) > 0:
      if len(inRange) == 0:
        self.inactivity = 0
        aStarAction = self.aStarPath.pop(0)
        self.recent_moves.append(aStarAction)
        if len(self.aStarPath) == 0 and closestDist < 5 and not agent.isPacman:
          self.findNewEntryPoint(gameState, agentLocation)
          self.debugDraw([agentLocation],[1,0,0])
          self.debugDraw([self.entry_point],[0,1,0])
        return aStarAction
      else:
        self.aStarPath = []

    
    if agent.isPacman or agentLocation == self.entry_point:
      self.atHome_count = 0
      self.entry_point = None
    elif not self.entry_point:
      self.atHome_count += 1

    if self.atHome_count > 15 and not self.entry_point:
      self.atHome_count = 0
      self.findNewEntryPoint(gameState, agentLocation)
      self.debugDraw([agentLocation],[1,0,0])
      self.debugDraw([self.entry_point],[0,1,0])

    
    if (self.inactivity > 80):
      self.is_going_home = True

    if agentLocation == self.entry_point:
      self.entry_point = None

    all_actions = gameState.getLegalActions(self.index)
    all_actions.remove(Directions.STOP)
    
    
    
    # if pacmans carry food, and faceing ghost chase them, 
    # pacmans will try to find distance to home to save food
    evaluations = util.Counter()
    if len(inRange) > 0:
      isGhostInRange = True
      dis = self.findDistanceToHome(agentLocation)
      if dis < 5 and closestDist < 7 and agent.numCarrying > 0:
        self.is_going_home = True
    
    reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if (reversed_direction in all_actions and 
        len(all_actions) > 1 and 
        len(inRange) == 0 and 
        agentLocation not in self.trap_list and
        agent.isPacman):
      # print('remove reverse '+str(reversed_direction))
      all_actions.remove(reversed_direction)
    elif len(all_actions) == 1 and agentLocation not in self.trap_list:
      self.trap_list.append(agentLocation)
      self.trap_depths[agentLocation] = 0

    to_rem = []
    neighbour_trap_depths = []
    max_val = float('-inf')
    for action in all_actions:
      path = [agentLocation]
      new_state = gameState.generateSuccessor(self.index, action)
      new_pos = new_state.getAgentPosition(self.index)
      if new_pos in self.capsules and len(all_actions) == 2 and len(inRange) > 0:
        return action
      if new_pos in self.trap_list:
        neighbour_trap_depths.append(self.trap_depths[new_pos])
      if new_pos == self.start_pos:
        to_rem.append(action)
        continue
      value, n = self.simulate(new_state, 3, path, None, isGhostInRange)
      if self.ghost_seen > -1:
        # in case a ghost is seen, start over the evaluation again
        break
      evaluations[action] = value #/n
      if value > max_val: max_val = value

    if self.ghost_seen > -1:
      max_val = float('-inf')
      for action in all_actions:
        path = [agentLocation]
        new_state = gameState.generateSuccessor(self.index, action)
        new_pos = new_state.getAgentPosition(self.index)
        if new_pos in self.capsules and len(all_actions) == 2 and len(inRange) > 0:
          return action
        if new_pos in self.trap_list:
          neighbour_trap_depths.append(self.trap_depths[new_pos])
        if new_pos == self.start_pos:
          to_rem.append(action)
          continue
        value, n = self.simulate(new_state, 3, path, None, isGhostInRange)
        evaluations[action] = value #/n
        if value > max_val: 
          max_val = value 

    if (agentLocation not in self.trap_list and 
        ((len(inRange) > 0 and len(neighbour_trap_depths) > 0 and 
        len(neighbour_trap_depths) == len(all_actions) - 1) or
        (len(neighbour_trap_depths) == len(all_actions)))):
      self.trap_list.append(agentLocation)
      self.trap_depths[agentLocation] = max(neighbour_trap_depths) + 1
        

    for action in to_rem:
      if action in all_actions:
        all_actions.remove(action)

    if len(all_actions) == 0 :
      best_action = Directions.STOP
    else :
      chosen_actions = []
      for action in all_actions:
        if evaluations[action] == max_val: 
          chosen_actions.append(action)
      # in case of more than one best action choose random action
      best_action = random.choice(chosen_actions)

    self.recent_moves.append(best_action)
    #if there is feature, choose best actions with max value
    if self.isStuck() and len(evaluations) > 1:
      self.stuck_counter = 0
      del evaluations[best_action]
      self.recent_moves.pop()
      best_action = evaluations.argMax()
      self.recent_moves.append(best_action)

      if not agent.isPacman and len(inRange) > 0:
        self.findNewEntryPoint(gameState, agentLocation)
        self.debugDraw([agentLocation],[1,0,0])
        self.debugDraw([self.entry_point],[0,1,0])

    if (agent.numCarrying - self.numCarrying) == 0 and agent.isPacman:
      self.inactivity += 1
    elif agent.numCarrying > 0:
      self.inactivity = 0

    return best_action

class ProtectingAgent(ReflexCaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  home_x = None
  escape_pos = []
  detour_no_entry = []
  _lastSeenEnmy = []
  _lastSeenInvader = []
  _chase = False
  _foodDefendingCentroids = None
  _aStarTarget = None
  _foodDefendingDict = None
  _foodDefending = None
  _patrolIndex = True
  _isAttacked = False
  _reSetPos = []
  lastEatenFood = None
  previousDefendingFood = None
  currentDefendingFood = None
  middleCentrePoints = []
  _notvisted_food = []

  def aStarSearch(self, gameState, myPos, goal_pos):
    """overwrite a timer functionality, if exceed timer, return None"""
    time_start = time.time()

    myPQ = util.PriorityQueue()
    startPos = myPos
    startNode = (startPos, '',0, [])
    myPQ.push(startNode,self.heuristic(startPos,startPos))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
      node = myPQ.pop()
      curPos, action, cost, path = node
      if (not curPos in visited) or cost < best_g.get(curPos):
        visited.add(curPos)
        best_g[curPos]=cost
        if self.isGoalState(curPos, goal_pos):
          path = path + [(curPos, action)]
          actions = [action[1] for action in path]
          del actions[0]
          return actions
        all_actions = self.possible_directions

        for succAction in all_actions:
          succ_x, succ_y = Actions.getSuccessor(curPos, succAction)
          succPos = (int(succ_x), int(succ_y))
          if (gameState.hasWall(int(succ_x), int(succ_y)) or 
            succPos in self.detour_no_entry): continue
          newNode = (succPos, succAction, cost + 1, path + [(node, action)])
          myPQ.push(newNode,self.heuristic(succPos,startPos)+cost+1)

      time_elapsed = time.time() - time_start
      if (time_elapsed > 0.9):
        return None

  def findCenterPos(self, gameState):
    total_width = gameState.data.layout.width
    total_height = gameState.data.layout.height
    offset = 0
    detour_offset = -1
    if gameState.isOnRedTeam(self.index):
      offset = -1
      detour_offset = 0
    # print('finding center pos')
    
    if not self.home_x:
      self.home_x = m.floor(total_width / 2)
    for y in range(total_height):
      if not gameState.hasWall(self.home_x + offset, y) :
        self.escape_pos.append((self.home_x + offset, y))
      if not gameState.hasWall(self.home_x + detour_offset, y) :
        self.detour_no_entry.append((self.home_x + detour_offset, y))

    return

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.start_pos = gameState.getAgentPosition(self.index)
    self._foodDefending = self.getFoodYouAreDefending(gameState).asList()
    self.findCenterPos(gameState)

  def _controller(self, gameState):
    # observe enemy and the enemy is attacking home
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    futureInvader = [a for a in enemies if ((not a.isPacman) and a.getPosition() != None)]

    currentDefendingFood = self.getFoodYouAreDefending(gameState).asList()
    previousDefendingFood = self._foodDefending
    # agents share vision
    if len(invaders) > 0:

      self._chase = True

    # ghost eats a pacman/ pacman return home:
    if len(self._lastSeenInvader) > 0 and len(invaders) <len(self._lastSeenInvader):
        self._isAttacked = False
        self._chase = False
        self._reSetPos = gameState.getAgentPosition(self.index)
        self._patrolIndex = 0

    # ghost is eaten by pacman
    if (gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index)):
      self._chase = False
      self._isAttacked = False
      self._reSetPos = gameState.getAgentPosition(self.index)
      self._patrolIndex = 0

    self._lastSeenEnmy = enemies
    self._lastSeenInvader = invaders

  def chooseAction(self, gameState):

    self._controller(gameState)

    """ when defender return to Initial position. meaning that they are either
     1) just got eaten or 2) just respawn. 

    either situation, ghost goes to the middle of entries to the other side of map
    once ghost reach one of the entry points, patrol to another entry points and keep
    patrolling untill observed there's food missing in our side of map.

    """
    if self._chase == False:
      prevState = self.getPreviousObservation()
      currentDefendingFood = self.getFoodYouAreDefending(gameState).asList()
      previousDefendingFood = currentDefendingFood

      if prevState != None:
        previousDefendingFood = self.getFoodYouAreDefending(prevState).asList()

      # if detect food is missing, flag attacked
      if len(currentDefendingFood) < len(previousDefendingFood):
        self._isAttacked = True
        eatenFood = set(previousDefendingFood) - set(currentDefendingFood)
        eatenFood = list(eatenFood)
        self.lastEatenFood = eatenFood
        self._notvisted_food = currentDefendingFood
        #  go to the last eaten food first
        if len(eatenFood) == 1:
          self._aStarTarget = eatenFood[0]
      # if defender dont' realized is attacked, patrol around the middle of the centre points
      elif not self._isAttacked:
          heightPartition = 3
          total_height = gameState.data.layout.height
          # middle
          uppderSection =  m.floor(((heightPartition-1)/heightPartition)*(total_height))
          lowerSection = m.floor((1/heightPartition)*(total_height))
          middleCentrePoints = [(x,y) for (x,y) in self.escape_pos if (y<=uppderSection and y>=lowerSection)]

          self.middleCentrePoints = middleCentrePoints
          self._aStarTarget = middleCentrePoints[self._patrolIndex]
          
          if gameState.getAgentPosition(self.index) == middleCentrePoints[self._patrolIndex]:            
            if self._patrolIndex+1 >= len(middleCentrePoints):
              nextPatrolIndex = 0
            else:
              nextPatrolIndex = self._patrolIndex +1
            self._patrolIndex = nextPatrolIndex
            self._aStarTarget = middleCentrePoints[nextPatrolIndex]


      # in the very rare cases that defending ghost get to the last eaten food, but pacman still havn't eaten anything
      # then the ghost is navigated to the centre points again and patrolling
      # this is very bad stratigical design, could be improved later
      if (gameState.getAgentPosition(self.index) == self._aStarTarget):
        if (gameState.getAgentPosition(self.index) == self.lastEatenFood[0]):
          print("problem head" +str(self.middleCentrePoints[self._patrolIndex]))
          # goes to the next food that is closest
          # distance = [self.getMazeDistance(self.lastEatenFood[0], food) for food in self._notvisted_food]
          # self._notvisted_food = np.argMin(distance)

          self._aStarTarget = self.middleCentrePoints[self._patrolIndex]

        elif gameState.getAgentPosition(self.index) == self.middleCentrePoints[self._patrolIndex]:
          if self._patrolIndex+1 >= len(self.middleCentrePoints):
            nextPatrolIndex = 0
          else:
            nextPatrolIndex = self._patrolIndex +1
            
          self._patrolIndex = nextPatrolIndex
          self._aStarTarget = self.middleCentrePoints[nextPatrolIndex]

      actions = self.aStarSearch(gameState, gameState.getAgentPosition(self.index), self._aStarTarget)

      if actions != None:
        if (len(actions)>0):
          return actions[0]
        else:
          # this should not be happen, unless there are unprecedented test cases
          Actions = gameState.getLegalActions(self.index)
          return(random.choice(Actions))
          print("shouldn't happen")
          pring(gameState.getAgentState(self.index))

      # In case a* exceed the timer, choose the action that has minimal distance to the goal
      else:
        dists = []
        print("illegal actions")
        Actions = gameState.getLegalActions(self.index)
        for action in Actions:
          nextState = gameState.generateSuccessor(self.index, action)
          nextPos = nextState.getAgentPosition(self.index)
          dists.append(self.getMazeDistance(nextPos,self._aStarTarget))

        minDist = min(dists)
        bestActions = []
        for action,dist in zip(Actions,dists):
          if minDist == dist:
            bestActions.append(action)

        # print("aStar Fail"+ str(time_elapsed2))

        return random.choice(bestActions)

      # house keeping
      self._foodDefending = currentDefendingFood

    else:
      """
      Picks among the actions with the highest Q(s,a).
      """

      actions = gameState.getLegalActions(self.index)

      # You can profile your evaluation time by uncommenting these lines
      # start = time.time()
      values = [self.evaluate(gameState, a) for a in actions]
      # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      foodLeft = len(self.getFood(gameState).asList())

      if foodLeft <= 2:
        bestDist = 9999
        for action in actions:
          successor = self.getSuccessor(gameState, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start,pos2)
          if dist < bestDist:
            bestAction = action
            bestDist = dist
        return bestAction

      return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # if pacman is at respawn position and the previous state is not at the respawn, meaning that he's eaten
    # didn't consider ghost move to respawn point/ forcely by enemy pacman, like our ghost is scared, and by avoiding
    #  the enemy pacman, we move to respawn, but the result is traiv, no different than suicide and teleport back to
    # respawn
    if gameState.getAgentState(self.index).scaredTimer>0:


      # print(gameState.getAgentPosition(self.index))
      if gameState.getInitialAgentPosition(self.index) == myState.getPosition():
        features['isEaten'] = 1
        # print("behaving wierdly")

    return features

  def getWeights(self, gameState, action):

    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,'isEaten':-3000}




