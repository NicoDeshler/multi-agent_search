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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        point_gain = successorGameState.getScore() - currentGameState.getScore()  # points gained in successor state compared to current state
        curPos = currentGameState.getPacmanPosition()       # current pacman position
        curGhostStates = currentGameState.getGhostStates()  # current ghost states
        curDist2Ghosts = [manhattanDistance(curPos, ghostState.getPosition()) for ghostState in
                      curGhostStates]  # manhattan distances to ghosts in current state
        newDist2Ghosts = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in
                      newGhostStates]  # manhattan distances to ghosts in successor

        # 1-SPACE GHOST HORIZON
        A = 0   # Number of ghosts distance 1 away from Pacman in current state
        B = 0   # Number of ghosts Pacman touches after performing action to reach successor
        C = 0   # Number of ghosts distance 1 away from Pacman after performing action to reach successor
        for i in range(len(newGhostStates)):
            if curDist2Ghosts[i] == 1:
                A += 1
            if newDist2Ghosts[i] == 0:
                B+=1
            if newDist2Ghosts[i] == 1:
                C+=1

        # 1-SPACE FOOD HORIZON
        curFood = currentGameState.getFood()    # current food
        curDist2Food = [manhattanDistance(curPos, foodPos) for foodPos in curFood.asList()]  # manhattan distances to current food from current pacman position
        newDist2Food = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]  # manhattan distances to new food from successor pacman position
        D = 0   # Number of food dots of distance 1 from new state
        E = point_gain/9    # Normalized points gained from transition to new state
        for d in newDist2Food:
            if d == 1:
                D += 1

        # INF-SPACE HORIZON UPDATE
        # Add deal breaker: If all 1-Space Horizon values are 0, move towards the nearest food dot
        F = 0  # boolean for if the new position has a closer nearest dot than the current position
        if len(curDist2Food) == len(newDist2Food) > 0 and min(curDist2Food) > min(newDist2Food):
            F = 1

        # Future functionalities
        # use gameState.getWalls() to navigate the possibility of being boxed in on 3 sides

        # Combine values into ghost and point functions
        Fg = A + B + C
        Fp = D + 2*E + F

        # Return linear combo of ghost and point functions
        return -10*Fg + 100*Fp

        # naive implementation just greedily considers the score of the game in each successor
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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

        # Helper functions
        def min_value(gameState, agent_index, depth):
            valid_actions = gameState.getLegalActions(agent_index)
            successors = [gameState.generateSuccessor(agent_index, action) for action in valid_actions]
            # if no actions available, just evaluate current state
            if not valid_actions:
                return self.evaluationFunction(gameState)

            if agent_index == gameState.getNumAgents() - 1:
                # end of agent move cycle
                if depth == 1:
                    # at leaf nodes
                    successor_vals = [self.evaluationFunction(s) for s in successors]
                else:
                    # cycle back to pacman agent and decrement the depth
                    successor_vals = [max_value(s, 0, depth - 1) for s in successors]
            else:
                # explore next agent's actions
                successor_vals = [min_value(s, agent_index + 1, depth) for s in successors]

            return min(successor_vals)

        def max_value(gameState, agent_index, depth):
            valid_actions = gameState.getLegalActions(agent_index)
            successors = [gameState.generateSuccessor(agent_index, action) for action in valid_actions]
            # if no actions available, just evaluate current state
            if not valid_actions:
                return self.evaluationFunction(gameState)
            # get successor values from minizers
            successor_vals = [min_value(s, agent_index + 1, depth) for s in successors]
            return max(successor_vals)

        # Do root node
        valid_actions = gameState.getLegalActions(self.index)
        successors = [gameState.generateSuccessor(self.index, action) for action in valid_actions]
        successor_vals = [min_value(s, self.index + 1, self.depth) for s in successors]
        action_index = successor_vals.index(max(successor_vals))

        return valid_actions[action_index]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # helper function
        def alphabeta(state, agent_index, depth, alpha, beta):
            # Performs alpha-beta pruning on a minimax search tree with multiple adversaries

            # actions of current node
            valid_actions = state.getLegalActions(agent_index)

            # leaf nodes
            if depth == 0 or state.isWin() or state.isLose() or not valid_actions:
                return self.evaluationFunction(state), None

            # get next agent index
            next_agent_index = (agent_index + 1) % state.getNumAgents()

            # decrement depth if cycle through agents is complete
            if next_agent_index == 0:
                depth -= 1

            # maximizer (pacman)
            if agent_index == 0:
                # instantiate return objects
                value = float("-inf")
                action = valid_actions[0]

                for i in range(len(valid_actions)):
                    # set value to max of successors
                    next_value,_ = alphabeta(state.generateSuccessor(agent_index, valid_actions[i]),next_agent_index, depth, alpha, beta)
                    if next_value > value:
                        value = next_value
                        action = valid_actions[i]

                    # update alpha
                    alpha = max(alpha, next_value)

                    # short circuit for pruning
                    if alpha > beta:
                        break

            # minimizer (ghosts)
            else:
                # instantiate return objects
                value = float("inf")
                action = valid_actions[0]

                for i in range(len(valid_actions)):
                    # set value to min of successors
                    next_value, _ = alphabeta(state.generateSuccessor(agent_index, valid_actions[i]), next_agent_index, depth, alpha, beta)
                    if next_value < value:
                        value = next_value
                        action = valid_actions[i]

                    # update beta
                    beta = min(beta, next_value)

                    # short circuit for pruning
                    if beta < alpha:
                        break

            return value, action

        # outer call
        val, action = alphabeta(gameState, self.index, self.depth, float("-inf"), float("inf"))

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # expectimax helper
        def expectimax(state, agent_index, depth):

            # actions of current node
            valid_actions = state.getLegalActions(agent_index)

            # leaf nodes
            if depth == 0 or state.isWin() or state.isLose() or not valid_actions:
                return self.evaluationFunction(state), None

            # get next agent index
            next_agent_index = (agent_index + 1) % state.getNumAgents()

            # decrement depth if cycle through agents is complete
            if next_agent_index == 0:
                depth -= 1

            # maximizer (pacman)
            if agent_index == 0:
                # instantiate return objects
                value = float("-inf")
                action = valid_actions[0]

                for i in range(len(valid_actions)):
                    # set value to max of successors
                    next_value, _ = expectimax(state.generateSuccessor(agent_index, valid_actions[i]), next_agent_index,
                                              depth)
                    if next_value > value:
                        value = next_value
                        action = valid_actions[i]


            # chance agent (ghosts)
            else:
                # instantiate return objects
                value = 0
                action = None
                for i in range(len(valid_actions)):
                    # set value to min of successors
                    next_value, _ = expectimax(state.generateSuccessor(agent_index, valid_actions[i]), next_agent_index,
                                              depth)
                    value += next_value

                if valid_actions:
                    value = value/len(valid_actions)

            return value, action

        # outer call
        val, action = expectimax(gameState, self.index, self.depth)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return float("-inf")
    else:
        pos = currentGameState.getPacmanPosition()  # current pacman position
        Food = currentGameState.getFood().asList()
        FoodDistance = [manhattanDistance(pos, f) for f in Food]
        numCapsules = len(currentGameState.getCapsules())
        SumThreshedFoodDist = 0
        depth = 1 # should be equal to search depth of agent but unfortunately this is not accessible from gameState
        for f in FoodDistance:
            if f > depth:
                SumThreshedFoodDist += f

        F_pos = 0
        F_neg = - 20 * numCapsules

        if len(FoodDistance) == 1 and FoodDistance[0] == 1:
            return float("inf")

        if SumThreshedFoodDist > 0:
            F_pos = currentGameState.getScore() + 1 / SumThreshedFoodDist
        else:
            F_pos =  currentGameState.getScore()

        return F_pos + F_neg

        """       
        E = 0
        if Food:
            nearestFood = min([manhattanDistance(pos, f) for f in Food])
            if nearestFood > depth:
                E = 1 / nearestFood
        #return currentGameState.getScore() + E * 0.0001
        return currentGameState.getScore()
        """
    """
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return float("-inf")
    else:
        return currentGameState.getScore()
        
        Score = currentGameState.getScore()
        Food = currentGameState.getFood().asList()
        Capsules = currentGameState.getCapsules()
        GhostStates = currentGameState.getGhostStates()

        pos = currentGameState.getPacmanPosition()  # current pacman position
        #Dist2Ghosts = [manhattanDistance(Pos, ghostState.getPosition()) for ghostState in GhostStates]  # manhattan distances to ghosts in current state
        #ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]


        A = 0
        if pos in Capsules:
            A = 1

        B = len(Food)
        C = 0
        D = 0
        for gs in GhostStates:
            gs_pos = gs.getPosition
            gs_t = gs.scaredTimer
            d2gs = manhattanDistance(pos, gs.getPosition())
            if gs_t > d2gs:
                C += gs_t - d2gs
            else:
                D += 1/d2gs

        E = 0
        if Food:
            nearestFood = min([manhattanDistance(pos,f) for f in Food])
            if nearestFood > 2:
                E = nearestFood
            E = nearestFood

        Fpos = A + C
        Fneg = B + D + E

        #return Fpos - 100*Fneg
        return Score + 1000*(1/E)**2
        """
"""
        # 1-SPACE Horizon terms
        A = 0  # Number of ghosts distance 1 away from Pacman in current state

        for i in range(len(GhostStates)):
            if Dist2Ghosts[i] == 1:
                A += 1

        # 1-SPACE GHOST HORIZON
        A = 0  # Number of ghosts distance 1 away from Pacman in current state
        B = 0  # Number of ghosts Pacman touches in current state
        for i in range(len(GhostStates)):
            if Dist2Ghosts[i] == 1:
                A += 1
            if Dist2Ghosts[i] == 0:
                B += 1

        # 1-SPACE FOOD HORIZON
        Dist2Food = [manhattanDistance(pos, foodPos) for foodPos in
                        Food.asList()]  # manhattan distances to current food from current pacman position
        D = 0  # Number of food dots of distance 1 from new state
        #E = currentGameState.getScore()  # pacman score
        for d in Dist2Food:
            if d == 1:
                D += 1

        # INF-SPACE HORIZON UPDATE
        # Add deal breaker: move towards the nearest food dot
        F = 0  # boolean for if the new position has a closer nearest dot than the current position
        if len(Dist2Food) > 0:
            F = 1/min(Dist2Food)

        # Combine values into ghost and point functions
        Fg = A + B
#        Fp = D + 2 * E + F
        Fp = D + F

        # Return linear combo of ghost and point functions
        return -10 * Fg + 10 * Fp

"""

# Abbreviation
better = betterEvaluationFunction
