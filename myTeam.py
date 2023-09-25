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
import copy
from game import Actions
from game import Directions
import time
import math as m
import numpy as np
from util import nearestPoint, Counter
from util import manhattanDistance as mdist

SIGHT_RANGE = 5
NOISE = 6
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
   # part 1: copy paste
  ##############################################################################################
  agents = [eval(first)(firstIndex), eval(second)(secondIndex)]
  share = SharedInfo(agents)   # part 1
  return agents
  ##########
  # Agents #
  ##########

class ReflexCaptureAgent(CaptureAgent):
      
  recent_moves = []
  possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
  trap_exits = util.Counter()
  potential_traps = []

      
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

  def aStarSearch(self, gameState, myPos, goal_pos, avoid_x = -1):
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
          if (gameState.hasWall(int(succ_x), int(succ_y)) or succ_x == avoid_x): continue
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
    if gameState.isOnRedTeam(self.index):
      offset = -1
    
    if not self.home_x:
      self.home_x = m.floor(total_width / 2)

    for y in range(total_height):
      if not gameState.hasWall(self.home_x + offset, y) :
        self.escape_pos.append((self.home_x + offset, y))

    return

  def setTrapDetails(self, coordinate, gameState, legalNeighbours, step_count, trap_list):
    given_trap_list = copy.deepcopy(trap_list)
    food = None
    if coordinate in self.food: food = coordinate
    neighboursQ = []
    visitedQ = [coordinate]
    current_trap = []
    trap_exit = None
    # if coordinate not in self.trap_list:
    given_trap_list.append(coordinate)
    current_trap.append(coordinate)
    self.trap_depths[coordinate] = step_count
    for nei in legalNeighbours:
      neighboursQ.append(nei)
      steps = step_count
    while len(neighboursQ) > 0:
      next_nei = neighboursQ.pop(0)
      visitedQ.append(next_nei)
      if(next_nei in legalNeighbours):
        steps = step_count
      steps += 1

      neighbours = Actions.getLegalNeighbors(next_nei, gameState.getWalls())
      neighbours.remove(next_nei)

      if len(neighbours) > 2:
        if food:
          self.trapped_food.push(food, (steps * -1))
          for capsule in self.capsules:
            self.trapped_food_paths[capsule+food] = self.aStarSearch(gameState, capsule, food)

        trap_exit = next_nei

        if steps > self.max_trap_depth:
          self.max_trap_depth = steps
        self.potential_traps.append(next_nei)
        continue

      if next_nei not in given_trap_list:
        given_trap_list.append(next_nei)
        current_trap.append(next_nei)
        self.trap_depths[next_nei] = steps

      for nei in neighbours:
        if nei not in visitedQ and nei not in neighboursQ:
          neighboursQ.append(nei)

    if trap_exit and len(current_trap) > 0:
      for pos in current_trap:
        self.trap_exits[pos] = trap_exit
    
    return given_trap_list

  def getAllTraps(self, gameState, redTeam, trap_list):
    given_trap_list = copy.deepcopy(trap_list)
    total_height = gameState.data.layout.height - 1
    x_min = 1
    x_max = self.home_x

    if not redTeam:
      x_min = self.home_x
      x_max = gameState.data.layout.width - 1

    not_trap = []

    for y in range(1, total_height):
      for x in range(x_min, x_max):
        coordinate = (x, y)
        if coordinate in given_trap_list: continue
        if not gameState.hasWall(x, y) :
          legalNeighbours = Actions.getLegalNeighbors(coordinate, gameState.getWalls())
          legalNeighbours.remove(coordinate)
          if len(legalNeighbours) > 1:
            continue
          given_trap_list = self.setTrapDetails(coordinate, gameState, legalNeighbours, 0, given_trap_list)

    return given_trap_list

  # part 1: copy paste
  ##############################################################################################

  # # this function returns a tupple (flag, enemyIndex) -- this only look 2 steps back, not 4 steps!
  # # a flag signal our ghost eat a pacman: True is eaten, False is not eaten
  # # enemyIndex is the agent index that is eaten
  # # ---- implementation logics
  # # two conditions to make sure pacman is eaten
  # # 1. in the last observation, there's 1 pacman that is 1 or 2 distance away from any ghost agents
  # # 2. in the current observation, both agents can't observed such pacman any more
  def _1checkIfDefenderAtePacman(self, gameState, homeX):

    curState = gameState
    prevState = self.share.getLastObservation(self.index)

    offset = 0
    if gameState.isOnRedTeam(self.index):
      offset = -1
    escape_x = homeX + offset


    if prevState == None:
      return (False, None)
    else:
      # a tuple of (agentState, enemy index)
      curInvadersIndex = [i for i in self.getOpponents(curState)
                          if curState.getAgentState(i).isPacman
                          and curState.getAgentState(i).getPosition() != None]

      # a tuple of (agentState, enemy index)
      prevEnemies = [(prevState.getAgentState(i), i) for i in self.getOpponents(prevState)]
      prevInvaders = [a for a in prevEnemies if a[0].getPosition() != None]
      # potential defender
      prevDefenders = [(prevState.getAgentState(i), i) for i in self.getTeam(prevState) if not prevState.getAgentState(i).isPacman or
                        (prevState.getAgentState(i).isPacman and prevState.getAgentState(i).getPosition()[0] == escape_x)]

      # compute index of 1 distant invaders to any defender in previous state
      _1_2_distancePrevInvaders = []
      for prevInvader, prevInvaderIndex in prevInvaders:
        for prevDefender, prevDefenderIndex in prevDefenders:
          x,y = curState.getAgentPosition(prevDefenderIndex)

          if (self.getMazeDistance(prevInvader.getPosition(), prevDefender.getPosition()) == 1 and
                not prevDefender.scaredTimer>0):
            if prevInvader.isPacman:
              _1_2_distancePrevInvaders.append(prevInvaderIndex)
            # boundary cases
            elif x == homeX:
              _1_2_distancePrevInvaders.append(prevInvaderIndex)

          elif (self.getMazeDistance(prevInvader.getPosition(), prevDefender.getPosition()) == 2 and
                not prevDefender.scaredTimer>0):
              if prevInvader.isPacman:
                _1_2_distancePrevInvaders.append(prevInvaderIndex)
            # boundary cases
              # elif x == homeX:
              #   _1_2_distancePrevInvaders.append(prevInvaderIndex)

      # now check if we can observe invader index in our current state
      if len(_1_2_distancePrevInvaders)!= 0: 
        for prevInvaderIndex in _1_2_distancePrevInvaders:
            # a tuple of (agentState, enemy index)
            # check no ghost is eaten
            if prevInvaderIndex not in curInvadersIndex:
              return (True, prevInvaderIndex)
        return (False, None)
      # if distance is greater than 1, it's defnitely not eaten
      return (False, None)

  # # # this function retruns a flag signal the agent that called this method
  # # # is a defender/ghost and was eaten by invader/pacman when scared
  # # # two conditions to make sure our ghost is eaten
  # # # 1. in the last observation, our defender agent is scared
  # # # 2. in the current observation, our defender is at the initial getPosition
  # def _checkIfPacmanAteDefender(self, gameState, prevState):
  #   if (prevState.getAgentState(self.index).scaredTimer>0 and
  #        gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index)):
  #     return True
  #   else:
  #     return False


  # this method is to chekc if our pacman ate opponent's scare ghost
  def _3checkIfOurPacmanAteGhost(self, gameState):

    curState = gameState
    prevState = self.share.getLastObservation(self.index)

    if prevState == None:
      return (False, None)

    else:
      # a tuple of (agentState, enemy index)
      curDefendersIndex = [i for i in self.getOpponents(curState)
                          if not curState.getAgentState(i).isPacman
                          and curState.getAgentState(i).getPosition() != None]
      # a tuple of (agentState, enemy index)
      prevEnemies = [(prevState.getAgentState(i), i) for i in self.getOpponents(prevState)]
      prevDefenders = [a for a in prevEnemies if (not a[0].isPacman) and a[0].getPosition() != None]
      # a list of agentState
      prevInvaders = [prevState.getAgentState(i) for i in self.getTeam(prevState) if prevState.getAgentState(i).isPacman]
      # compute index of 1 distant invaders to any defender in previous state
      _1_2_distancePrevDefenders = []
      for prevDefender, prevDefenderIndex in prevDefenders:
        for prevInvader in prevInvaders:
          if (self.getMazeDistance(prevInvader.getPosition(), prevDefender.getPosition()) == 1):
            _1_2_distancePrevDefenders.append(prevDefenderIndex)
          elif (self.getMazeDistance(prevInvader.getPosition(), prevDefender.getPosition()) == 2):
            _1_2_distancePrevDefenders.append(prevDefenderIndex)
      # now check if we can observe invader index in our current state
      if len(_1_2_distancePrevDefenders)!= 0: 
        for prevDefenderIndex in _1_2_distancePrevDefenders:
          if prevDefenderIndex not in curDefendersIndex:
            # makes sure ghost is scared
            if prevState.getAgentState(prevDefenderIndex).scaredTimer>0:
              return (True, prevDefenderIndex)
        return (False, None)
      # if distance is greater than 1, it's defnitely not eaten
      return (False, None)

  # this method takes a gameState and compare gameState with the 1 last gameState backward
  # return a food or Capsule position, if there's food missing 
  def _checkIfFoodMissing(self,gameState):
    # G2 is the current state, G1 is 1 state back(what our other agent observe)
    G1 = self.share.getLastObservation(self.index)
    G2 = gameState

    if (G1 == None or
          G2 == None):
      return (False, None)
    else:
      food_G1 = self.getFoodYouAreDefending(G1).asList()
      food_G2 = self.getFoodYouAreDefending(G2).asList()
      capsule_G1 = self.getCapsulesYouAreDefending(G1)
      capsule_G2 = self.getCapsulesYouAreDefending(G2)


      # eaten food 1
      if len(food_G2) < len(food_G1):
        eatenFood = set(food_G1) - set(food_G2)
        eatenFood = list(eatenFood)
        #  go to the last eaten food first
        if len(eatenFood) == 1:
          return (True,eatenFood[0])
      # eaten food 1
      elif len(capsule_G2) < len(capsule_G1):
        eatenCapsule = set(capsule_G1) - set(capsule_G2)
        eatenCapsule = list(eatenCapsule)
        #  go to the last eaten food first
        if len(eatenCapsule) == 1:
          return (True,eatenCapsule[0])
      else:
        return (False,None)

  # don't need to know what this function do, it is a helper function
  def registerSharedInfo(self, share):
    self.share = share
  # this function has to be called everytime in chooseAction, it is updating self.share
  # to make other methods work
  def updateShareInfo(self,gameState):
    self.share.updateOppLocations(gameState, self.index)
    self.share.updatePrevState(gameState,self.index)
    # can comment display distirbution
    # self.displayDistributionsOverPositions([dist.copy() for dist in self.share.oppLocs.values() if dist])

class AttackingAgent(ReflexCaptureAgent):

  home_x = None
  capsules = []
  food = None
  start_pos = None
  aStarPath = []
  pathToAttackPos = util.Counter()
  deepest_trapped_food = None
  escape_pos = []
  food_count = None
  is_going_home = False
  stuck_counter = 0
  recent_moves = []
  trapped_food = util.PriorityQueue()
  inactivity = 0
  inactive_limit = 20
  atHome_count = 0
  trap_list = []
  trap_depths = util.Counter()
  trapped_food_paths = util.Counter()
  inTrapCount = 0
  ghost_seen = -1
  max_trap_depth = 0
  max_eval_value = 99999
  entry_point = None
  max_distance = 40
  numCarrying = 0
  eval_list = []
  simulation_depth = 3
  ghost_list = []
  isGhostScared = False
  random_move = 0
  potential_traps = []
  prev_pos = None

  def setGhostList(self, inRange):
    self.ghost_list = copy.deepcopy(inRange)

  def ghostDistToFood(self, food):
    minDist = min([self.getMazeDistance(food, ghost_pos) for ghost_pos in self.ghost_list])
    return minDist

  #position for foods that is not in trap
  def getUntrappedFood(self, gameState, food_list) :
    not_trapped_list = []
    for food in food_list:
      if food not in self.trap_list or self.ghostDistToFood(food) > 5:
        not_trapped_list.append(food)
    return not_trapped_list

  #This method handles the initial setup of the agent
  def registerInitialState(self, gameState):
    self.start_pos = gameState.getAgentPosition(self.index)
    # keeping previous position record in every step
    self.prev_pos = copy.deepcopy(self.start_pos)
    CaptureAgent.registerInitialState(self, gameState)
    # local record of capsule and food list and food count
    self.capsules = self.getCapsules(gameState)
    self.food = self.getFood(gameState).asList()
    self.food_count = len(self.food)

    self.findCenterPos(gameState)

    avoid_x = self.home_x
    if not gameState.isOnRedTeam(self.index):
      avoid_x = self.home_x - 1

    # finding path from each escape position to every other escape position
    for pos in self.escape_pos:
      for another_pos in self.escape_pos:
        if pos != another_pos:
          self.pathToAttackPos[pos+another_pos] = self.aStarSearch(gameState, pos, another_pos, avoid_x)

    # finding all possible traps in the opponents field
    self.trap_list = self.getAllTraps(gameState, (not gameState.isOnRedTeam(self.index)), self.trap_list)
    # making another round of search to find traps among identified potential trap positions
    for pos in self.potential_traps:
      legal_neighbours = Actions.getLegalNeighbors(pos, gameState.getWalls())
      legal_neighbours.remove(pos)
      neighbour_trap_depths = []
      for neighbour in legal_neighbours:
        if neighbour in self.trap_list:
          neighbour_trap_depths.append(self.trap_depths[neighbour])
      if (pos not in self.trap_list and 
          (len(neighbour_trap_depths) >= len(legal_neighbours) - 1)):
        self.trap_list = self.setTrapDetails(pos, gameState, legal_neighbours, max(neighbour_trap_depths) + 1, self.trap_list)
    self.share.initialise(gameState)  # this is new

  def findInRangeGhost(self, state, myPos) :
    """
    Compute distance to the nearest ghost
    and find if there are ghosts in range
    """
    inRange = []
    closestDist = float('inf')
    myAgent = state.getAgentState(self.index)
    enemy_indices = self.getOpponents(state)

    if len(self.ghost_list) > 0:
      inRange = copy.deepcopy(self.ghost_list)
      closestDist = min([self.getMazeDistance(myPos, ghost_pos) for ghost_pos in self.ghost_list])
      return inRange, closestDist

    self.isGhostScared = False
    for index in enemy_indices:
      enemy_agent = state.getAgentState(index)
      enemy_pos = enemy_agent.getPosition()
      if (enemy_pos and self.getMazeDistance(myPos, enemy_pos) < 7 and
          ((not enemy_agent.isPacman and enemy_agent.scaredTimer < 7) or 
          (enemy_agent.isPacman and myAgent.scaredTimer > 0))):
        inRange.append(enemy_pos)
        if self.getMazeDistance(myPos, enemy_pos) < closestDist:
          closestDist = self.getMazeDistance(myPos, enemy_pos)
          if enemy_agent.scaredTimer > 0: self.isGhostScared = True
          else: self.isGhostScared = False
    return inRange, closestDist

  def getFeatures(self, state, action):
    features = util.Counter()
    successor = state.generateSuccessor(self.index, action)
    myPos = successor.getAgentState(self.index).getPosition()

    # Compute score from successor state
    features['successorScore'] = self.getScore(successor)    
    # in general give preference to being a pacman
    features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

    features['inTrapCount'] = self.inTrapCount

    distanceToHome = min([self.getMazeDistance(myPos, pos) for pos in self.escape_pos])
    inRange, closestDist = self.findInRangeGhost(successor, myPos)

    if closestDist == float('inf'):
        closestDist = self.max_distance

    if self.entry_point:
      """
      if entry_point has value, it would indicate pacman requires diversion to go out.
      pacman is encouraged to go out.
      """
      features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0
      features['distanceToEntry'] = self.getMazeDistance(myPos, self.entry_point)
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
      foodList = self.getUntrappedFood(successor, foodList)
      if (len(foodList) == 0 and 
        successor.getAgentState(self.index).numCarrying == 0):
        foodList = self.food

    elif (successor.getAgentState(self.index).isPacman and 
        (self.numCarrying < (self.food_count * (1 / 10)))):
      self.is_going_home = False

    #returns position of the nearest capsule  
    if len(self.capsules) > 0:
      minDistance = 0
      if (myPos not in self.capsules):
        minDistance = min([self.getMazeDistance(myPos, food) for food in self.capsules])
      features['distanceToCapsule'] = minDistance

    if (self.numCarrying == 0 and successor.getAgentState(self.index).isPacman):
      features['distanceToHome'] = distanceToHome

    # returns distance to nearest Food
    if (len(foodList) > 0):
      minDistance = 0
      if (myPos not in foodList):
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
    ghost_on_timer = False

    enemy_indices = self.getOpponents(state)
    for index in enemy_indices:
      enemy_agent = state.getAgentState(index)
      if (not enemy_agent.isPacman and enemy_agent.scaredTimer > 6):
        ghost_on_timer = True

    weights['isPacman'] = 2
    weights['location_trapped'] = -3
    weights['trap_depth'] = 5
    weights['distanceToGhost'] = 6
    weights['successorScore'] = 2
    weights['distanceToFood'] = -15
    weights['distanceToCapsule'] = -3
    weights['inTrapCount'] = 0

    distanceToHome = min([self.getMazeDistance(myPos, pos) for pos in self.escape_pos])
    inRange, closestDist = self.findInRangeGhost(successor, myPos)
    if len(inRange) > 0 or self.ghost_seen > -1:
      weights['distanceToFood'] = -6
      weights['distanceToCapsule'] = -6
      weights['location_trapped'] = -5
      # weights['trap_depth'] = 5
      if (self.numCarrying < 4):
        # (self.numCarrying == 0 and distanceToHome < 5) or 
        weights['distanceToGhost'] = 5
        weights['distanceToCapsule'] = -15
        # weights['distanceToFood'] = -15
        weights['isPacman'] = 5
      # if (not successor.getAgentState(self.index).isPacman):
      #   weights['distanceToGhost'] = 1
      if myPos in self.trap_list:
        weights['distanceToGhost'] = -8
        weights['distanceToFood'] = -1
    elif ((not ghost_on_timer and myPos in self.trap_list) and 
      (self.inTrapCount > 7 or closestDist < 6)):
      weights['inTrapCount'] = 5 * self.trap_depths[myPos]

    if self.entry_point:
      weights['distanceToEntry'] = -10
      weights['isPacman'] = 5
      weights['distanceToGhost'] = 10

    if self.numCarrying == 0 and successor.getAgentState(self.index).isPacman:
      weights['distanceToHome'] = 5

    if self.is_going_home:
      weights['distanceToHome'] = -2
      weights['distanceToFood'] = -1
      weights['distanceToCapsule'] = -1
    return weights

  def evaluate(self, state, action):
    """
    rewards and penalty are shaped in terms of features and weights
    The features and weights vectors are multiplied here to return rewards
    """
    features = self.getFeatures(state, action)
    weights = self.getWeights(state, action)
    if (features['distanceToFood'] == 0 and 
        features['trap_depth'] == 0 and
        features['distanceToGhost'] > 5):
      pos = state.getAgentState(self.index).getPosition()
      legal_actions = state.getLegalActions(self.index)
      legal_actions.remove(Directions.STOP)
      legal_action = legal_actions[0]
      predecessor = state.generateSuccessor(self.index, legal_action)
      predPos = predecessor.getAgentState(self.index).getPosition()
      if predPos not in self.trap_list:
        return self.max_eval_value
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

  def simulate(self, state, depth, path, parent_value, isGhostInRange, path_value):
    """
    Simulates positions at a given depth
    """
    mypos = state.getAgentPosition(self.index)
    path.append(mypos)
    food_list = self.getFood(state).asList()
    inRange, closestDist = self.findInRangeGhost(state, mypos)
    if (len(inRange) > 0 and len(self.ghost_list) == 0):
      self.setGhostList(inRange)
      self.ghost_seen = closestDist

    self_eval = self.evaluate(state, Directions.STOP)
    path_value.append(self_eval / ((self.simulation_depth + 1) - depth))
    if isGhostInRange and state.getAgentState(self.index).isPacman:
      # if parent could see and current location is capsule, then capsule is important
      if mypos in self.capsules:
        path_value.pop()
        path_value.append(self.max_eval_value)
        self.eval_list.append(copy.deepcopy(path_value))
        return (self.max_eval_value)

    if(parent_value!=None and self_eval < parent_value):
      # if parent's value is not none and current value has dropped from parent's value
      _, parent_ghost_dist = self.findInRangeGhost(state, path[-2])
      if parent_ghost_dist > closestDist:
        # return own value if ghost can be seen here
        self.eval_list.append(copy.deepcopy(path_value))
        return (self_eval / ((self.simulation_depth + 1) - depth))
      else:
        # return parent's higher value otherwise, assuming the drop is 
        # due to reasons like going away from food
        value_to_set = copy.deepcopy(path_value)
        value_to_set.pop()
        self.eval_list.append(value_to_set)
        return (parent_value / ((self.simulation_depth + 1) - (depth + 1)))

    if not isGhostInRange and len(inRange) > 0:
      isGhostInRange = True

    if mypos in food_list or mypos in self.ghost_list or depth == 0:
      # if current position is in either food list or ghost list or the depth of search is over
      # return own value
      self.eval_list.append(copy.deepcopy(path_value))
      return (self_eval / ((self.simulation_depth + 1) - depth)) 
  
    # simulate child positions
    legal_actions = state.getLegalActions(self.index)
    legal_actions.remove(Directions.STOP)

    ## Preventing agent from taking reverse direction
    reversed_direction = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
    if (reversed_direction in legal_actions and
        len(legal_actions) > 1 and
        state.getAgentState(self.index).isPacman):
      legal_actions.remove(reversed_direction)
    elif len(legal_actions) == 1 and mypos not in self.trap_list:
      self.trap_list.append(mypos)
      self.trap_depths[mypos] = 0


    val = []
    to_rem = []
    neighbour_trap_depths = []
    for action in legal_actions:
      state_prime = state.generateSuccessor(self.index, action)
      pos_prime = state_prime.getAgentPosition(self.index)
      if pos_prime in self.trap_list:
        neighbour_trap_depths.append(self.trap_depths[pos_prime])
      if pos_prime in self.capsules and isGhostInRange:
        v = self.max_eval_value
        continue
      if((pos_prime in path) or (pos_prime == self.start_pos)):
        # remove loops in path
        # remove any possibility to be eaten by ghost
        to_rem.append(action)
        continue
      v = self.simulate(state_prime, depth - 1, path, self_eval, isGhostInRange, path_value)
      path.pop()
      path_value.pop()
      val.append(v)

      if (mypos not in self.trap_list and state.getAgentState(self.index).isPacman and 
        len(neighbour_trap_depths) == len(legal_actions)):
        self.trap_list.append(mypos)
        self.trap_depths[mypos] = max(neighbour_trap_depths) + 1

    for rem_action in to_rem:
      legal_actions.remove(rem_action)

    if(len(val) == 0):
      self.eval_list.append(copy.deepcopy(path_value))
      return (self_eval / ((self.simulation_depth + 1) - depth))

    return max(val)

  #the agent try to find new entry point when start next attack
  def findNewEntryPoint(self, gameState, myPos):
    maxDist = 0
    for pos in self.escape_pos:
      dist = self.getMazeDistance(myPos, pos)
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

  def getValues(self):
    max_length = max([len(_list) for _list in self.eval_list])
    min_length = min([len(_list) for _list in self.eval_list])

    while max_length != min_length:
      min_val = float('inf')
      min_index = -1
      mini_length = float('inf')
      for i in range(len(self.eval_list)):
        _list = self.eval_list[i]
        # if 
        if (_list[min_length-1] < min_val or 
          (_list[min_length-1] == min_val and len(_list) < mini_length)):
          min_val = _list[min_length-1]
          min_index = i
          mini_length = len(_list)
      self.eval_list.pop(min_index)
      max_length = max([len(_list) for _list in self.eval_list])
      min_length = min([len(_list) for _list in self.eval_list])

    if len(self.eval_list) == 1: return self.eval_list[0]

    index = -1

    while index >= -max_length and len(self.eval_list) > 1:
      max_val = max([_list[index] for _list in self.eval_list])
      min_index = []
      for i in range(len(self.eval_list)):
        _list = self.eval_list[i]
        if _list[index] < max_val:
          min_index.append(i)
      for rem_index in reversed(min_index):
        self.eval_list.pop(rem_index)
      index -= 1

    return self.eval_list[0]

  def getBestActions(self, evaluation, all_actions):
    actions = copy.deepcopy(all_actions)
    eval_list = []
    for action in actions:
      eval_list.append(evaluation[action])

    max_length = max([len(_list) for _list in eval_list])
    min_length = min([len(_list) for _list in eval_list])

    index = 0

    while index < max_length and len(eval_list) > 1:
      min_indices = []
      for i in range(len(eval_list)):
        _list = eval_list[i]
        if len(_list) == index:
          min_indices.append(i)
      for rem_index in reversed(min_indices):
        eval_list.pop(rem_index)
        actions.pop(rem_index)
      max_val = max([_list[index] for _list in eval_list])
      min_indices = []  
      for i in range(len(eval_list)):
        _list = eval_list[i]
        if _list[index] < max_val:
          min_indices.append(i)
      for rem_index in reversed(min_indices):
        eval_list.pop(rem_index)
        actions.pop(rem_index)
      index += 1
      max_length = max([len(_list) for _list in eval_list])

    return actions
    
  def getRandomAction(self, gameState):
    all_actions = gameState.getLegalActions(self.index)
    all_actions.remove(Directions.STOP)

    self.random_move += 1

    if self.random_move > 6:
      self.random_move = 0
      return None

    no_entry_x = self.home_x - 1
    if gameState.isOnRedTeam(self.index):
      no_entry_x = self.home_x

    to_rem = None
    for action in all_actions:
      new_state = gameState.generateSuccessor(self.index, action)
      new_pos_x, new_pos_y = new_state.getAgentPosition(self.index)
      if (new_pos_x == no_entry_x or (new_pos_x, new_pos_y) == self.start_pos): 
        to_rem = action

    if to_rem: all_actions.remove(to_rem)
    
    reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if (reversed_direction in all_actions and 
        len(all_actions) > 1):
      all_actions.remove(reversed_direction)

    return random.choice(all_actions)

  def getNearFoodCount(self, myPos):
    food_count = 0
    
    for food_pos in self.food:
      if self.getMazeDistance(myPos, food_pos) < 8:
        food_count += 1

    return food_count

  def chooseAction(self, gameState):

    agent = gameState.getAgentState(self.index)
    agentLocation = gameState.getAgentPosition(self.index)

    if self.getMazeDistance(agentLocation, self.prev_pos) > 1:
      self.aStarPath = []
    self.prev_pos = agentLocation

    # if pacman on his own side, so reset the num carrying timer
    self.ghost_seen = -1
    self.ghost_list = []
    self.numCarrying = agent.numCarrying

    if self.entry_point and agentLocation in self.escape_pos and len(self.aStarPath) == 0:
      if self.entry_point != agentLocation:
        self.aStarPath = copy.deepcopy(self.pathToAttackPos[agentLocation+self.entry_point])
    # pacman is starting from the start position
    if agentLocation == self.start_pos:
      toss = random.randint(-1, 0)
      self.entry_point = self.escape_pos[toss]

    # after pacman eats the deepest trapped food of the moment
    if agentLocation in self.trap_list:
      self.inTrapCount += 1
    else:
      self.inTrapCount = 0


    if agentLocation == self.deepest_trapped_food:
      self.deepest_trapped_food = None

    if not agent.isPacman:
      self.food = self.getFood(gameState).asList()
      food_not_trapped = []
      for food in self.food:
        if food not in self.trap_list:
          food_not_trapped.append(food)
      if len(food_not_trapped) < 4 :
        self.inactive_limit = 40
      self.is_going_home = False
      self.food_count = len(self.food)
      self.inactivity = 0

    if (agentLocation in self.capsules) :
      self.capsules.remove(agentLocation)
      if (not ((agent.numCarrying > (self.food_count * (9 / 10))) or
        (agent.numCarrying > 0 and gameState.data.timeleft < 80))):
        self.is_going_home = False
      near_food_count = self.getNearFoodCount(agentLocation)
      if near_food_count < 4 :
        while (not self.trapped_food.isEmpty() and not self.deepest_trapped_food):
          self.deepest_trapped_food = self.trapped_food.pop()
          self.aStarPath = copy.deepcopy(self.trapped_food_paths[agentLocation+self.deepest_trapped_food])
          if (self.deepest_trapped_food not in self.food or 
            self.aStarPath == 0 or 
            self.getMazeDistance(agentLocation, self.deepest_trapped_food) > 10):
            self.deepest_trapped_food = None
            self.aStarPath = []

    if agentLocation in self.food:
      self.food.remove(agentLocation)
      self.inactivity = 0
    elif agent.isPacman:
      self.inactivity += 1

    isGhostInRange = False
    inRange, closestDist = self.findInRangeGhost(gameState, agentLocation)

     # the agent will select new path to follow when start new attack
    
    if len(self.aStarPath) > 0:
      if len(inRange) > 0 and agent.isPacman:
        self.aStarPath = []
      else:
        self.inactivity = 0
        aStarAction = self.aStarPath.pop(0)
        self.recent_moves.append(aStarAction)
        return aStarAction


    if ((agent.numCarrying > (self.food_count * (1 / 3)) and 
        len(inRange) > 0 and closestDist < 8) or 
        (agent.numCarrying > (self.food_count * (9 / 10))) or
        (agent.numCarrying > 0 and gameState.data.timeleft < 80)):
      self.is_going_home = True

    
    if agent.isPacman or agentLocation == self.entry_point:
      self.atHome_count = 0
      self.entry_point = None
    elif not self.entry_point:
      self.atHome_count += 1

    if self.atHome_count > 15 and not self.entry_point:
      self.atHome_count = 0
      self.findNewEntryPoint(gameState, agentLocation)
      if agentLocation in self.escape_pos:
        self.aStarPath = copy.deepcopy(self.pathToAttackPos[agentLocation+self.entry_point])
        return self.aStarPath.pop(0)
      self.debugDraw([agentLocation],[1,0,0])
      self.debugDraw([self.entry_point],[0,1,0])
    
    if (self.inactivity > self.inactive_limit):
      self.is_going_home = True

    all_actions = gameState.getLegalActions(self.index)
    all_actions.remove(Directions.STOP)

    if (self.random_move > 0):
      action = self.getRandomAction(gameState)
      if action: return action
      self.findNewEntryPoint(gameState, agentLocation)
      if agentLocation in self.escape_pos:
        self.aStarPath = copy.deepcopy(self.pathToAttackPos[agentLocation+self.entry_point])
        return self.aStarPath.pop(0)
      self.debugDraw([agentLocation],[1,0,0])
      self.debugDraw([self.entry_point],[0,1,0])
    
    # if pacmans carry food, and faceing ghost chase them, 
    # pacmans will try to find distance to home to save food
    evaluations = util.Counter()
    if len(inRange) > 0:
      isGhostInRange = True
      self.ghost_seen = closestDist
      self.setGhostList(inRange)
      dis = self.findDistanceToHome(agentLocation)
      if dis < 5 and closestDist < 7 and agent.numCarrying > 4:
        self.is_going_home = True

      if (not agent.isPacman and closestDist < 3 and
        not self.entry_point):
        return self.getRandomAction(gameState)
    
    reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if (reversed_direction in all_actions and 
        len(all_actions) > 1 and 
        len(inRange) == 0 and 
        agentLocation not in self.trap_list and
        agent.isPacman):
      all_actions.remove(reversed_direction)
    elif reversed_direction not in all_actions: reversed_direction = None

    to_rem = []
    neighbour_trap_depths = []
    if agentLocation in self.trap_list and self.simulation_depth == 3:
      self.simulation_depth = self.trap_depths[agentLocation]
      if self.simulation_depth < 3:
        self.simulation_depth = 3
      if self.simulation_depth > 6:
        self.simulation_depth = 6
    else:
      self.simulation_depth = 3
    
    if self.ghost_seen == -1:
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
        value = self.simulate(new_state, self.simulation_depth, path, None, isGhostInRange, [])
        if self.ghost_seen > -1:
          # in case a ghost is seen, start over the evaluation again
          self.eval_list = []
          break
        evaluations[action] = self.getValues()
        self.eval_list = []

    if self.ghost_seen > -1:
      to_rem = []
      neighbour_trap_depths = []
      if (reversed_direction != None and reversed_direction not in all_actions):
        all_actions.append(reversed_direction)
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
        value = self.simulate(new_state, self.simulation_depth, path, None, isGhostInRange, [])
        evaluations[action] = self.getValues()
        self.eval_list = []

    if (agentLocation not in self.trap_list and 
        ((len(inRange) > 0 and len(neighbour_trap_depths) > 0 and 
        len(neighbour_trap_depths) == len(all_actions) - 1) or
        (len(neighbour_trap_depths) == len(all_actions)))):
      self.trap_list.append(agentLocation)
      self.trap_depths[agentLocation] = max(neighbour_trap_depths) + 1
        

    for action in to_rem:
      if action in all_actions:
        all_actions.remove(action)

    best_action = Directions.STOP
    if len(all_actions) > 0:
      best_action_list = self.getBestActions(evaluations, all_actions)
      best_action = random.choice(best_action_list)

    self.recent_moves.append(best_action)
    #if there is feature, choose best actions with max value
    if self.isStuck() and len(evaluations) > 1:
      self.stuck_counter = 0
      del evaluations[best_action]
      all_actions.remove(best_action)
      self.recent_moves.pop()
      if len(all_actions) == 0 :
        best_action = Directions.STOP
      elif len(all_actions) > 0:
        best_action_list = self.getBestActions(evaluations, all_actions)
        best_action = random.choice(best_action_list)

      self.recent_moves.append(best_action)

      if not agent.isPacman and len(inRange) > 0:
        self.findNewEntryPoint(gameState, agentLocation)
        if agentLocation in self.escape_pos:
          self.aStarPath = copy.deepcopy(self.pathToAttackPos[agentLocation+self.entry_point])
          return self.aStarPath.pop(0)
        self.debugDraw([agentLocation],[1,0,0])
        self.debugDraw([self.entry_point],[0,1,0])

    new_state = gameState.generateSuccessor(self.index, best_action)
    next_step_agent = new_state.getAgentState(self.index)
    if (agent.isPacman and not next_step_agent.isPacman):
      self.random_move = 1
    if (closestDist == 2 and not self.isGhostScared and 
      not all(move == Directions.STOP for move in self.recent_moves[-4:])):
      _, new_closestDist = self.findInRangeGhost(new_state, new_state.getAgentPosition(self.index))
      if new_closestDist == 1: 
        best_action = Directions.STOP
        self.recent_moves.pop()
        self.recent_moves.append(best_action)

    # all_actions = gameState.getLegalActions(self.index)
    # if best_action not in all_actions: 
    #   best_action = Directions.STOP
    

    return best_action

class ProtectingAgent(ReflexCaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  home_x = None
  escape_pos = []
  # used in A* for defender
  # prevent going to other enemy's map
  detour_no_entry = []

  # controller variables
  _lastSeenInvader = []
  _chase = False
  _aStarTarget = None
  _foodDefending = None
  _patrolIndex = True
  _isAttacked = False
  _reSetPos = []
  lastEatenFood = None
  previousDefendingFood = None
  currentDefendingFood = None
  middleCentrePoints = []
  _notvisted_food = []

  # trap list
  trap_list = []
  trap_depths = util.Counter()
  trapped_food = util.PriorityQueue()
  trapped_food_paths = util.Counter()
  max_trap_depth = 0

  # invader in trap
  invaderInTrap = False


  # this A * is overriden because it is preventing to go to the enemy map
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

  # this findCenterPos is overriden because it append detour_no_entry
  def findCenterPos(self, gameState):
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
      if not gameState.hasWall(self.home_x + detour_offset, y) :
        self.detour_no_entry.append((self.home_x + detour_offset, y))

    return

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.start_pos = gameState.getAgentPosition(self.index)
    self._foodDefending = self.getFoodYouAreDefending(gameState).asList()
    self.findCenterPos(gameState)
    self.capsules = self.getCapsules(gameState)
    self.getMiddle(gameState)
    self.share.initialise(gameState)  # this is new

    # trap List function
    self.food = self.getFood(gameState).asList()    
    # self.attack_path_counter += 1
    self.trap_list = self.getAllTraps(gameState, (gameState.isOnRedTeam(self.index)), self.trap_list)
    self.food_count = len(self.food)
    for pos in self.potential_traps:
      legal_neighbours = Actions.getLegalNeighbors(pos, gameState.getWalls())
      legal_neighbours.remove(pos)
      neighbour_trap_depths = []
      for neighbour in legal_neighbours:
        if neighbour in self.trap_list:
          neighbour_trap_depths.append(self.trap_depths[neighbour])
      if (pos not in self.trap_list and 
          (len(neighbour_trap_depths) >= len(legal_neighbours) - 1)):
        # self.trap_list.append(pos)
        # self.trap_depths[pos] = max(neighbour_trap_depths) + 1
        self.trap_list = self.setTrapDetails(pos, gameState, legal_neighbours, max(neighbour_trap_depths) + 1, self.trap_list)


  def _controller(self, gameState):
    # observe enemy and the enemy is attacking home
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    # invaderIndex = [i for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition()!=None]

    currentDefendingFood = self.getFoodYouAreDefending(gameState).asList()
    previousDefendingFood = self._foodDefending
    # agents share vision
    if len(invaders) > 0:
      self._chase = True

    # ghost eats a pacman/ pacman return home:
    if len(self._lastSeenInvader) > 0 and len(invaders) <len(self._lastSeenInvader):    # and not self.invaderInTrap:
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
      self.invaderInTrap = False

    self._lastSeenEnmy = enemies
    self._lastSeenInvader = invaders


    # check if Invader is in trap

    # for invader, index in zip(invaders,invaderIndex):
    #   invader_pos = invader.getPosition()
    #   if (invader_pos in self.trap_list) and (gameState.getAgentPosition(self.index) == self.trap_exits[invader_pos]):
    #     self.stopPos = gameState.getAgentPosition(self.index)
    #     self.trappedInvader = 
    #     self.invaderInTrap = True

    # if gameState.getAgentState(self.index).scaredTimer>0:
    #   self.invaderInTrap = False
  def getMiddle(self, gameState):
    heightPartition = 3
    total_height = gameState.data.layout.height
    # middle
    uppderSection =  m.floor(((heightPartition-1)/heightPartition)*(total_height))
    lowerSection = m.floor((1/heightPartition)*(total_height))
    middleCentrePoints = [(x,y) for (x,y) in self.escape_pos if (y<=uppderSection and y>=lowerSection)]
    self.middleCentrePoints = middleCentrePoints

  def chooseAction(self, gameState):

    self.updateShareInfo(gameState) # new

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

          self._aStarTarget = self.middleCentrePoints[self._patrolIndex]
          
          if gameState.getAgentPosition(self.index) == self.middleCentrePoints[self._patrolIndex]:            
            self._patrolIndex = (self._patrolIndex +1) % len(self.middleCentrePoints)
            self._aStarTarget = self.middleCentrePoints[self._patrolIndex]


      # in the very rare cases that defending ghost get to the last eaten food, but pacman still havn't eaten anything
      # then the ghost is navigated to the centre points again and patrolling
      # this is very bad stratigical design, could be improved later
      if (gameState.getAgentPosition(self.index) == self._aStarTarget):
        if (gameState.getAgentPosition(self.index) == self.lastEatenFood[0]):
          # goes to the next food that is closest
          # distance = [self.getMazeDistance(self.lastEatenFood[0], food) for food in self._notvisted_food]
          # self._notvisted_food = np.argMin(distance)
          self._aStarTarget = self.middleCentrePoints[self._patrolIndex]

        elif gameState.getAgentPosition(self.index) == self.middleCentrePoints[self._patrolIndex]:
          self._patrolIndex = (self._patrolIndex +1) % len(self.middleCentrePoints)
          self._aStarTarget = self.middleCentrePoints[self._patrolIndex]

      actions = self.aStarSearch(gameState, gameState.getAgentPosition(self.index), self._aStarTarget)

      if actions != None:
        if (len(actions)>0):
          return actions[0]
        else:
          # this should not be happen, unless there are unprecedented test cases
          Actions = gameState.getLegalActions(self.index)
          return(random.choice(Actions))
      

      # # In case a* exceed the timer, choose the action that has minimal distance to the goal
      # else:
      #   dists = []
      #   
      #   Actions = gameState.getLegalActions(self.index)
      #   for action in Actions:
      #     nextState = gameState.generateSuccessor(self.index, action)
      #     nextPos = nextState.getAgentPosition(self.index)
      #     dists.append(self.getMazeDistance(nextPos,self._aStarTarget))

      #   minDist = min(dists)
      #   bestActions = []
      #   for action,dist in zip(Actions,dists):
      #     if minDist == dist:
      #       bestActions.append(action)

      #   

      #   return random.choice(bestActions)

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
      

      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      
      if self.invaderInTrap and gameState.getAgentPosition(self.index) == self.stopPos:
        return Directions.STOP

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
    successor = self.getSuccessor(gameState, action)
    features = self.getFeatures(gameState, successor, action)
    weights = self.getWeights(gameState, successor, action)
    return features * weights

  def getFeatures(self, gameState, successor, action):
    features = util.Counter()
    curPos = gameState.getAgentPosition(self.index)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    # set the closest to be the chased enemy, however, no tie break here
    # chased_invader = min(invaders, key = lambda invader: self.getMazeDistance(invader, myPos))

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0
    # for the puprpose of eating enemy
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    # stop action
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    # revert action
    if action == rev: features['reverse'] = 1

    # for now I set the chased_invader has the minimal distance to defender
    if invaders != []:
      chased_invader = min(invaders, key = lambda invader: self.getMazeDistance(invader.getPosition(), myPos))    
      # when ghost is scared, add the incentive to move to the grid that is at the distance of 2
      # in addition, prefer the grid that has more legalNeighbours, prefer 2 legalNeighbours than 1 legal NEIGHBOURS

      if (gameState.getAgentState(self.index).scaredTimer>0 or
            any([self.getMazeDistance(chased_invader.getPosition(), capsule) == 1 for capsule in self.capsules])):
        
        legalNeighbours = Actions.getLegalNeighbors(myPos,gameState.getWalls())
        legalNeighbours.remove(myPos)
        timeleft = gameState.getAgentState(self.index).scaredTimer
        
        # avoid
        if self.getMazeDistance(curPos, gameState.getInitialAgentPosition(self.index)) > timeleft:
          if self.getMazeDistance(myPos, chased_invader.getPosition())==2:
            features['avoid'] = 0.5
            if not myPos in self.trap_list:
              features['avoid'] = 1
        # suicide
        else:
          
          features['avoid'] = 0
          if myPos==gameState.getInitialAgentPosition(self.index):
            features['avoid'] = -1

    return features

  def getWeights(self, gameState,successor, action):

    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,'avoid':3000}

class SharedInfo():

  # the constructor is called in creatTeam(), in the constructor, a share object will
  # have two objects of agent and inside of each agent, it will have this same share object
  # please be very careful with pointers
  def __init__(self, agents):
    self.agents = agents
    for agent in self.agents:
      agent.registerSharedInfo(self)
    # this is because agents is a list has length only 2
    self.myAgentIndex = {0:0,1:0,2:1,3:1}
    self.otherAgentIndex = {0:2,1:3,2:0,3:1}
    
  # this method takes a gameState object and initialize information from gameState to sharedInfo
  # oppLocs dictionary stores a distribution of possible enemy's location
  # preState store a previous observation(state) of the two agents 
  def initialise(self, gameState):
    self.walls = gameState.getWalls()
    # a dictionary of dictionary
    # outer dictionary map opponent agent ID into
    # a innter dictionary of a location of distribution
    self.oppLocs = {}
    for oppIndex in self.agents[0].getOpponents(gameState):
      self.oppLocs[oppIndex] = Counter({gameState.getInitialAgentPosition(oppIndex): 1})
    self.numAgents = gameState.getNumAgents()
    # a dictionary that keep previous state
    self.prevStates = {}
    self.agentsAtePacman = {}

  def updateOppLocations(self, gameState, agentIndex):

    # move is the index of previous moved agent, only update moved agent
    moved = (agentIndex + self.numAgents - 1) % self.numAgents
    # a list of noisy observation with index
    agentDists = gameState.getAgentDistances()
    locDict = {index: gameState.getAgentPosition(index) for index in range(self.numAgents)}
    currLoc = locDict[agentIndex]
    # location of my own two agents
    ownLocs = [locDict[agent.index] for agent in self.agents]

    # check if their pacman died
    FL1, index1 = self.updateIfDefendersAtePacman(gameState, agentIndex)
    FL2, foodPos = self.agents[self.myAgentIndex[agentIndex]]._checkIfFoodMissing(gameState)
    FL3, index2 = self.agents[self.myAgentIndex[agentIndex]]._3checkIfOurPacmanAteGhost(gameState)

    for oppIndex, oldLocs in self.oppLocs.items():
      pos = gameState.getAgentPosition(oppIndex)
      # if the opponent is seen
      if pos is not None:
        self.oppLocs[oppIndex] = Counter({pos: 1})
        continue
      # if the enemy pacman just died OR enemy ghost die because our pacman ate scared ghost
      elif oppIndex == index1 or oppIndex == index2:
        if ((agentIndex ==0 and oppIndex == 1) or
              (agentIndex==1 and oppIndex==2) or
              (agentIndex==2 and oppIndex==3) or
              (agentIndex==3 and oppIndex ==0)):
          self.oppLocs[oppIndex] = Counter({gameState.getInitialAgentPosition(oppIndex): 1})
          continue
        elif((agentIndex == 0 and oppIndex == 3) or
              (agentIndex==1 and oppIndex==0) or
              (agentIndex==2 and oppIndex==1) or
              (agentIndex==3 and oppIndex == 2)):

          newLocs = Counter()
          x,y = gameState.getInitialAgentPosition(oppIndex)
          cands=[(x,y)]
          for u, v in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if not self.walls[u][v]:
              cands.append((u, v))
          prob = 1 / len(cands)
          for loc in cands:
            newLocs[loc] = prob
          self.oppLocs[oppIndex] = newLocs
          continue

      # if a food just disappeared
      elif FL2 == True:
        if (oppIndex == moved):
          newCands = []
          for x, y in oldLocs:
            newCands.append((x,y))
            # enumerate all possible locations that opponent may moved to
            for u, v in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
              if not self.walls[u][v]:
                newCands.append((u, v))
          if foodPos in newCands:
            self.oppLocs[oppIndex] = Counter({foodPos: 1})
            continue            
        else:
          if foodPos in [*oldLocs]:
            self.oppLocs[oppIndex] = Counter({foodPos: 1})
            continue

      newLocs = Counter()
      if (oppIndex == moved):
        for x, y in oldLocs:
          cands = [(x, y)]
          # enumerate all possible locations that opponent may moved to
          for u, v in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if not self.walls[u][v]:
              cands.append((u, v))
          oldprob = self.oppLocs[oppIndex][(x,y)]
          prob = oldprob / len(cands)
          for loc in cands:
            newLocs[loc] += prob


      # this is for opponent agent that hasn't made a move before our agent
      elif oppIndex != moved:
        newLocs = oldLocs

      # count the total probability after filtering 
      total = 0
      for loc, prob in newLocs.items():
        if self.isLocPossible(loc, ownLocs, currLoc, agentDists[oppIndex]):
          total += prob
        # filtering out impossible locations
        else:
          newLocs[loc] = 0
      # reassign a distribution now

      self.oppLocs[oppIndex] = Counter()
      for loc, prob in newLocs.items():
        if prob != 0:
          self.oppLocs[oppIndex][loc] = prob / total
      # when bug happens/ debugging print
      #if(self.oppLocs[oppIndex]=={}):
        
       #for loc, prob in newLocs.items():
          
        # pdb.set_trace()
      for loc, prob in newLocs.items():
        if prob != 0:
          self.oppLocs[oppIndex][loc] = prob / total

  def isLocPossible(self, loc, ownLocs, currLoc, ndist):
    x, y = loc
    # all() makes sure filter out the location that is in sight
    return (all(mdist(loc, ownLoc) > SIGHT_RANGE for ownLoc in ownLocs)
            and abs(ndist - mdist(currLoc, loc)) <= NOISE)
  # prevState will have two None object at the begining because observationHistory
  # is empty list in the begining
  def updatePrevState(self, gameState, agentIndex):
    # agent.observationHistory is empty list at the start
    agent = self.agents[self.myAgentIndex[agentIndex]]
    if len(agent.observationHistory) > 0:
      self.prevStates[agentIndex] = gameState

  def updateIfDefendersAtePacman(self, gameState, agentIndex):
    self.agentsAtePacman[agentIndex] = (False, None)
    agent = self.agents[self.myAgentIndex[agentIndex]]
    FL1, index = agent._1checkIfDefenderAtePacman(gameState, agent.home_x)
    self.agentsAtePacman[agentIndex] = (FL1, index)
    return FL1, index

  def getOppLocations(self, oppAgentIndex):
    return self.oppLocs[oppAgentIndex].copy()
  # this agent return 
  # return a copy of previous state, just to be safe
  def getLastObservation(self,agentIndex):
      if self.otherAgentIndex[agentIndex] in self.prevStates.keys():
        return self.prevStates[self.otherAgentIndex[agentIndex]]        
      else:
        return None
