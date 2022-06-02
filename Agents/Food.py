class Food:
    # def __init__(self, coordinates, food_type='A', hidden=0, reward=1):
    def __init__(self, coordinates, food_type='A', hidden=False, reward=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.food_type = food_type
        self.hidden = hidden
        self.reward = reward
        # self.color = color

    def is_hidden(self):
        # return self.hidden > 0
        return self.hidden

    def collect(self, food_hidden):
        # self.hidden = food_hidden
        self.hidden = True
        return self.reward

    # def run_hidden(self):
    #     self.hidden -= 1
    #     self.hidden = 0 if self.hidden <= 0 else self.hidden
    #     # self.hidden = not self.hidden
    #     return self.hidden

    def visible(self):
        self.hidden = False