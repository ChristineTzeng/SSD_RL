##Test
class GameState:
    def __init__(self, prevState=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState != None:  # Initial state
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def initialize(self, foods, agents):
        """
        Creates an initial game state.
        """
        self.data.initialize(foods, agents)

    def deepCopy(self):
        state = GameState(self)
        state.data = self.data.deepCopy()
        return state

    def getNumAgents(self):
        return len(self.data.agentStates)

    def getScore(self):
        return float(self.data.score)

    def getNumFood(self):
        return self.data.foods.count()

    def hasFood(self, x, y):
        return self.data.foods[x][y]

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data

    def __str__(self):

        return str(self.data)


class GameStateData:
    def __init__(self, prevState=None):
        """
        Generates a new data packet by copying information from its predecessor.
        """
        if prevState != None:
            self.foods = prevState.foods[:]
            self.agents = self.copyAgentStates(prevState.agents)
            self._eaten = prevState._eaten
            self.score = prevState.score

        self.foods = []
        self._foodAdded = None
        self._foodCollected = None
        self._agentMoved = None
        self.score = 0
        self.scoreChange = 0

    def initialize(self, foods, agents):
        """
        Creates an initial game state
        """
        self.foods = foods[:]
        #self.capsules = []
        self.score = 0
        self.scoreChange = 0

        self.agents = agents[:]

    def deepCopy(self):
        state = GameStateData(self)
        state.food = self.food.deepCopy()
        state._agentMoved = self._agentMoved
        state._foodCollected = self._foodCollected
        state._foodAdded = self._foodAdded
        return state

    def copyAgentStates(self, agents):
        copiedStates = []
        for agent in agents:
            copiedStates.append(agent.copy())
        return copiedStates

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        if other == None:
            return False
        # TODO Check for type of other
        if not self.agentStates == other.agentStates:
            return False
        if not self.food == other.food:
            return False
        if not self.capsules == other.capsules:
            return False
        if not self.score == other.score:
            return False
        return True

    def __str__(self):
        width, height = MAP_WIDTH, MAP_HEIGHT
        map = Grid(width, height)
        if isinstance(self.food, type((1, 2))):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agentState in self.agentStates:
            if agentState == None:
                continue
            if agentState.configuration == None:
                continue
            x, y = [int(i) for i in nearestPoint(agentState.configuration.pos)]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr(agent_dir)
            else:
                map[x][y] = self._ghostStr(agent_dir)

        for x, y in self.capsules:
            map[x][y] = 'o'

        return str(map) + ("\nScore: %d\n" % self.score)