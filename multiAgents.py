# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        evaluation = 0 
        currentGhostStates = currentGameState.getGhostStates()
        
        oldScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
        oldFood = currentGameState.getFood() 
        old_min = float('inf')
        
        currentPos = currentGameState.getPacmanPosition()
        
        for food in oldFood.asList():
            distance = manhattanDistance(currentPos, food)
            if  distance < old_min:
                old_min = distance
            
        
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            if distance == 1:
                evaluation += 1
                
            elif 3 >= distance > 1:
                evaluation += 0.5
                
            elif distance < old_min:
                evaluation += 0.5
            

        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) == 1:
                if ghost.scaredTimer > 1:
                    evaluation += 10
                else:
                    evaluation -= 9999
                
        if max(newScaredTimes) > max(oldScaredTimes):
            evaluation += 5
        
        
        evaluation_score = successorGameState.getScore() + evaluation
        return evaluation_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def getbestaction(state : GameState):
            
            actions = state.getLegalActions(0)
            max_value = float('-inf')
            
            for action in actions:
                action_v = value(state.generateSuccessor(0, action), 0, 1)
                
                if action_v[1] > max_value:
                    max_value = action_v[1]
                    chosen_action = action
                    
                
 
            return chosen_action
            
            
        def value(state : GameState, depth, agent_index):
            
            if agent_index == state.getNumAgents():
                agent_index = 0
                depth += 1
                            
            
            if (depth == self.depth or state.isWin() or state.isLose()):
                return ["", self.evaluationFunction(state)]
            
            elif agent_index == 0:
                return maxvalue(state, depth, agent_index)
            
            else:
                return minvalue(state, depth, agent_index)
            

            
            
        def maxvalue(state : GameState, depth, agent_index):
            
            v = ["",float('-inf')]
            actions = state.getLegalActions(agent_index)
            
            if not actions:
                return ["", self.evaluationFunction(state)]
            
            for action in actions:
                next_value = value(state.generateSuccessor(agent_index, action), depth, agent_index + 1)
                action_value = next_value[1]
                v = [action, max(v[1], action_value)] 
            
            return v
        
        
        def minvalue(state : GameState, depth, agent_index):
            
            v = ["",float('inf')]
            actions = state.getLegalActions(agent_index) 
            
            if not actions:
                return ["", self.evaluationFunction(state)]
            
            for action in actions:
                next_value = value(state.generateSuccessor(agent_index, action), depth, agent_index + 1)
                action_value = next_value[1]
                v = [action, min(v[1], action_value)]
            
            return v
        
            
        return getbestaction(gameState)
               
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def getbestaction(state : GameState):
            
            actions = state.getLegalActions(0)
            max_value = float('-inf')
            
            alpha = float('-inf')
            beta = float('inf')
            
            for action in actions:
                action_v = value(state.generateSuccessor(0, action), 0, 1, alpha, beta)
                
                if action_v[1] > max_value:
                    max_value = action_v[1]
                    chosen_action = action
                    
                alpha = max(alpha, max_value)
                    
                    
 
            return chosen_action
            
            
        def value(state : GameState, depth, agent_index, alpha, beta):
            
            if agent_index == state.getNumAgents():
                agent_index = 0
                depth += 1
                            
            
            if (depth == self.depth or state.isWin() or state.isLose()):
                return ["", self.evaluationFunction(state)]
            
            elif agent_index == 0:
                return maxvalue(state, depth, agent_index, alpha, beta)
            
            else:
                return minvalue(state, depth, agent_index, alpha, beta)
            

            
            
        def maxvalue(state : GameState, depth, agent_index, alpha, beta):
            
            v = ["",float('-inf')]
            actions = state.getLegalActions(agent_index)
            
            if not actions:
                return ["", self.evaluationFunction(state)]
            
            for action in actions:
                next_value = value(state.generateSuccessor(agent_index, action), depth, agent_index + 1, alpha, beta)
                action_value = next_value[1]
                v = [action, max(v[1], action_value)] 
                
                if v[1] >= beta:
                    return v
                alpha = max(alpha, v[1])
            
            return v
        
        
        def minvalue(state : GameState, depth, agent_index, alpha, beta):
            
            v = ["",float('inf')]
            actions = state.getLegalActions(agent_index) 
            
            if not actions:
                return ["", self.evaluationFunction(state)]
            
            for action in actions:
                next_value = value(state.generateSuccessor(agent_index, action), depth, agent_index + 1, alpha, beta)
                action_value = next_value[1]
                v = [action, min(v[1], action_value)]
                
                if v[1] <= alpha:
                    return v
                beta = min(beta, v[1])
            
            return v
        
            
        return getbestaction(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        actions = gameState.getLegalActions(0)
        max_value = float('-inf')
        chosen_action = ""

        for action in actions:
            action_value = self.expectimax_value(gameState.generateSuccessor(0, action), 0, 1)
            
            if action_value > max_value:
                max_value = action_value
                chosen_action = action

        return chosen_action

    def expectimax_value(self, state, depth, agent_index):
        if agent_index == state.getNumAgents():
            agent_index = 0
            depth += 1

        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agent_index == 0:
            return self.max_value(state, depth, agent_index)
        else:
            return self.expect_value(state, depth, agent_index)


    def max_value(self, state, depth, agent_index):
        v = float('-inf')
        actions = state.getLegalActions(agent_index)

        if not actions:
            return self.evaluationFunction(state)

        for action in actions:
            successor = state.generateSuccessor(agent_index, action)
            v = max(v, self.expectimax_value(successor, depth, agent_index + 1))

        return v

    def expect_value(self, state, depth, agent_index):
        v = 0.0
        actions = state.getLegalActions(agent_index)

        if not actions:
            return self.evaluationFunction(state)

        probability = 1.0 / len(actions)

        for action in actions:
            successor = state.generateSuccessor(agent_index, action)
            v += probability * self.expectimax_value(successor, depth, agent_index + 1)

        return v

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    activeGhosts = [] # Keep active ghosts(can eat pacman)
    scaredGhosts = [] # Keep scared ghosts(pacman should eat them for extra points)
    totalCapsules = len(currentGameState.getCapsules()) # Keep total capsules
    totalFood = len(food) # Keep total remaining food
    myEval = 0 # Evaluation value

    # Fix active and scared ghosts #
    for ghost in ghosts:
        if ghost.scaredTimer: # Is scared ghost
            scaredGhosts.append(ghost)
        else:
            activeGhosts.append(ghost)


    myEval +=  1.5 * currentGameState.getScore()
    myEval += -10 * totalFood
    myEval += -30 * totalCapsules

    # Keep distances from food, active and scared ghosts #
    foodDistances = []
    activeGhostsDistances = []
    scaredGhostsDistances = []

    # Find distances #
    for item in food:
        foodDistances.append(manhattanDistance(pacmanPosition,item))

    for item in activeGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition,item.getPosition()))

    for item in scaredGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition,item.getPosition()))

    # Fix evaluation based on food distances  #
    # It is very bad for pacman to have close #
    # food. He must eat it.                   #
    # Close food weight: -1                   #
    # Quite close food weight: -0.5           #
    # Far away food weight: -0.2              #
    for item in foodDistances:
        if item < 3:
            myEval += -1 * item
        if item < 7:
            myEval += -0.5 * item
        else:
            myEval += -0.2 * item

    # Fix evaluation based on scared ghosts distances    #
    # It is very bad for pacman to have close scared     #
    # ghosts. He must eat them so as to gain many points #
    # We should prefer to eat a ghost rather than eat a  #
    # close food                                         #
    # Close scared ghosts weight: -20                    #
    # Quite close scared ghosts weight: -10              #
    for item in scaredGhostsDistances:
        if item < 3:
            myEval -= 20 * item
        else:
            myEval -= 10 * item

    # Fix evaluation base on active ghosts distances    #
    # Pacman should avoid active ghosts                 #
    # Close ghost weight: 3                             #
    # Quite close ghost weight: 2                       #
    # Far away ghosts weight: 0.5                       #
    # We should prefer ghosts remaining far away        #
    for item in activeGhostsDistances:
        if item < 2:
            myEval -= 100 * item
        elif item < 6:
            myEval -= 2 * item
        else:
            myEval -= 1 * item

    return myEval

    

# Abbreviation
better = betterEvaluationFunction
