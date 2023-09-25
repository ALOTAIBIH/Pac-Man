# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
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
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  # def __init__(self, index):
  #   CaptureAgent.__init__(self, index)
  #   self.lastFoodEaten = None
  #   self.lastFood = []
  #   self.currentFoodList = []
  #   self.destination = None
  #   self.nonZeroCount = 0
  lastFoodEaten = None
  lastFood = []
  currentFoodList = []
  destination = None
  nonZeroCount = 0

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.findDefensePosition(gameState)
    self.lastFoodEaten = None
    self.lastFood = []
    self.currentFoodList = []
    self.destination = None
    self.nonZeroCount = 0
    
  #finding all the open positions and setting the mid position as the defense area
  def findDefensePosition(self,gameState):
    self.defensePositions = []
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    middleX = int((width-2)/2)
    if not self.red:
      middleX +=1
    middleY = int((height-2)/2)
    for i in range(1,height-1):
      if not gameState.hasWall(middleX,i):
        self.defensePositions.append((middleX,i))
    if len(self.defensePositions)<=2:
      self.firstDefensePosition = self.defensePositions[0]
    else:
      self.firstDefensePosition = self.defensePositions[int(len(self.defensePositions)/2)]

  #returns the position of nearst enemy  
  def getClosestEnemyPacman(self,gameState):
    opponents = self.getOpponents(gameState)
    dists = []
    opponentPos = []
    closestEnemies = []
    closestEnemy = None
    
    pacmen = []
    minDis =99999999
    myPos = gameState.getAgentState(self.index).getPosition()
    for index in self.getOpponents(gameState):
      opponent = gameState.getAgentState(index)
      if  opponent.isPacman  and opponent.getPosition() != None:                
          pacmen.append(opponent.getPosition())
    #when enemy is nearby
    if len(pacmen)>0:
      for pos in pacmen:
        dis = self.getMazeDistance(myPos,pos)
        if dis <minDis:
          minDis = dis
          closestEnemies.append(pos)
      closestEnemy = closestEnemies[-1]


    else:
      #when enemy's position is not known, use the position of food eaten
      if len(self.lastFood)>0 and len(self.currentFoodList)<len(self.lastFood):
        foodEaten = set(self.lastFood) - set(self.currentFoodList)
        closestEnemy = foodEaten.pop()
    return closestEnemy

    

  def chooseAction(self,gameState):
    self.currentFoodList = self.getFoodYouAreDefending(gameState).asList()
    self.myPos = gameState.getAgentPosition(self.index)
    #if reached target, set target as none
    if self.myPos == self.destination:
      self.destination = None
    #gets closest enemy
    closestEnemy = self.getClosestEnemyPacman(gameState)
   #go to closest enemy position
    if closestEnemy!=None:
      self.destination = closestEnemy
     #if closest enemy position is none and there are only few food left, go near food/capsule left
    if self.destination==None:
      if len(self.currentFoodList)< 4:
        self.destination = random.choice(self.currentFoodList + self.getCapsulesYouAreDefending(gameState))
      else:
        self.destination = self.firstDefensePosition


    #dont become pacman and dont reverse if possible
    goodActions = []
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    back = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if back in actions :
      actions.remove(back)

    for action in actions:
      nextState = gameState.generateSuccessor(self.index,action)
      if not nextState.getAgentState(self.index).isPacman:
        goodActions.append(action)
    
    if len(goodActions)==0:
      self.nonZeroCount = 0
    else:
      self.nonZeroCount +=1
    # if there is no option but to go back, use reverse
    if self.nonZeroCount==0 or self.nonZeroCount>4:
      goodActions.append(back)


    #going to destination using maze distance
    dists = []
    for action in goodActions:
      nextState = gameState.generateSuccessor(self.index,action)
      nextPos = nextState.getAgentPosition(self.index)
      if not nextState.getAgentState(self.index).isPacman:
        dists.append(self.getMazeDistance(nextPos,self.destination))
    minDist = min(dists)
    bestActions = []
    for action,dist in zip(goodActions,dists):
      if minDist == dist:
        bestActions.append(action)
    self.lastFood = self.currentFoodList
    return random.choice(bestActions)

    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
