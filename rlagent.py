import ast
import random
from agent import Agent


class RLAgent(Agent):

    def __init__(self, gamma=0.9, alpha=1, epsilon=1):
        super().__init__()
        self.rewards = self.get_rewards()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = {(x, y): {"north": 0, "south": 0, "east": 0, "west": 0}
                        for x in range(self.maxCoord[0]) for y in range(self.maxCoord[1])}
        self.targets = self.getTargets()

    def getTargets(self):
        ''' Return the targets defined in the world.'''
        msg = self.c.execute("info", "targets")
        res = ast.literal_eval(msg)
        # test
        # print('Received targets:', res)
        return res

    def target_reached(self):
        current_position = self.current_position()
        return bool(self.targets[current_position[0]][current_position[1]])

    def get_rewards(self):
        msg = self.c.execute("info", "rewards")
        rewards = ast.literal_eval(msg)
        # test
        print('Received map of obstacles:', rewards)
        return rewards

    def reward_at(self, coords):
        return self.rewards[coords[0]][coords[1]]

    def mark_visited(self, coords, color="coral"):
        super().mark_visited(coords, color)

    def goal_reached(self):
        return self.current_position() == self.goalNodePos

    def execute_random_action(self, no_turning_back=True):
        current_position = self.current_position()

        directions = [direction for direction in ("north", "south", "east", "west")
                      if self.is_visitable(*self.step(current_position, direction))]
        if all((no_turning_back,
                len(directions) > 1,
                self.opposite_direction(self.last_direction) in directions)):
            directions.remove(self.opposite_direction(self.last_direction))

        if random.random() < self.epsilon:
            direction = random.choice(directions)
        else:
            direction = max(self.q_table[current_position], key=self.q_table[current_position].get)

        self.c.execute("command", direction)
        self.last_direction = direction
        return direction

    def walk_randomly_until_goal_reached(self, episodic, no_turning_back=True):
        path = []
        while not self.goal_reached() and not self.target_reached():
            position_before_action = self.current_position()
            action = self.execute_random_action(no_turning_back)
            position_after_action = self.current_position()
            reward = self.reward_at(position_after_action)

            path.append((position_before_action, position_after_action, action, reward))
            if not episodic:
                self.update_q_table(position_before_action, position_after_action, action, reward)

            self.mark_visited(position_after_action)

        return path

    def go_home(self):
        self.c.execute("command", "home")
        self.last_direction = None
        self.mark_visited(self.current_position())

    def clean_path(self, path):
        for coord in set(path):
            self.unmark(coord[0])

    def update_q_table(self, previous_state, current_state, action, reward):
        q_of_s_a = self.q_table[previous_state][action]
        self.q_table[previous_state][action] = (
            q_of_s_a + self.alpha * (reward + self.gamma * max(self.q_table[current_state].values()) - q_of_s_a)
        )

    def update_q_table_from_path(self, path):
        for step in reversed(path):
            previous_state, current_state, action, reward = step
            self.update_q_table(previous_state, current_state, action, reward)

    def mark_arrow(self, coords, direction):
        self.c.execute("marrow", "{type},{row},{column}".format(type=direction, column=coords[0], row=coords[1]))

    def display_policy(self):
        for entry in self.q_table.items():
            state, directions = entry
            max_direction = max(directions.items(), key=lambda x: x[1])[0]
            if directions[max_direction] != 0:
                self.mark_arrow(state, max_direction)

    def q_learn(self, episodes, episodic=True, no_turning_back=True):
        for episode in range(episodes):
            path = self.walk_randomly_until_goal_reached(
                episodic,
                no_turning_back
            )
            self.clean_path(path)
            if episodic:
                self.update_q_table_from_path(path)
            self.go_home()
        for e in self.q_table.items(): print(e)


# STARTING THE PROGRAM:
def main():
    print("Starting client!")
    agent = RLAgent(gamma=0.9, alpha=1, epsilon=1)
    if agent.get_connection() != -1:
        agent.q_learn(
            episodes=10,
            episodic=False,
            no_turning_back=False
        )
        agent.display_policy()
    input()


if __name__ == "__main__":
    main()