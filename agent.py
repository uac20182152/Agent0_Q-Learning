import client
import ast
import random


class Agent:
    def __init__(self):
        self.c = client.Client('127.0.0.1', 50001)
        self.res = self.c.connect()
        random.seed()  # To become true random, a different seed is used! (clock time)
        self.weightMap = []
        self.maxCoord = self.get_max_coord()
        self.opposite_directions = {"north": "south", "south": "north", "east": "west", "west": "east"}

        self.obstacles = self.get_obstacles()
        self.direction_offset = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        self.goalNodePos = self.get_goal_position()
        self.last_direction = None

    def get_connection(self):
        return self.res

    def get_goal_position(self):
        msg = self.c.execute("info", "goal")
        goal = ast.literal_eval(msg)
        # test
        print('Goal is located at:', goal)
        return goal

    def current_position(self):
        msg = self.c.execute("info", "position")
        pos = ast.literal_eval(msg)
        # test
        print('Received agent\'s position:', pos)
        return pos

    def get_max_coord(self):
        msg = self.c.execute("info", "maxcoord")
        max_coord = ast.literal_eval(msg)
        # test
        print('Received maxcoord', max_coord)
        return max_coord

    def get_obstacles(self):
        msg = self.c.execute("info", "obstacles")
        obst = ast.literal_eval(msg)
        # test
        print('Received map of obstacles:', obst)
        return obst

    def is_visitable(self, x, y):
        return not self.obstacles[x][y]

    def step(self, pos, action):
        if action == "east":
            if pos[0] + 1 < self.maxCoord[0]:
                new_pos = (pos[0] + 1, pos[1])
            else:
                new_pos = (0, pos[1])

        elif action == "west":
            if pos[0] - 1 >= 0:
                new_pos = (pos[0] - 1, pos[1])
            else:
                new_pos = (self.maxCoord[0] - 1, pos[1])

        elif action == "north":
            if pos[1] - 1 >= 0:
                new_pos = (pos[0], pos[1] - 1)
            else:
                new_pos = (pos[0], 0)

        elif action == "south":
            if pos[1] + 1 < self.maxCoord[1]:
                new_pos = (pos[0], pos[1] + 1)
            else:
                new_pos = (pos[0], self.maxCoord[1] - 1)
        else:
            new_pos = None
        return new_pos

    def mark_visited(self, coords, color="coral"):
        self.c.execute("mark", str(coords)[1:-1].replace(" ", "") + "_" + color)

    def unmark(self, coords):
        # self.c.execute("mark_visited", str(node.getState())[1:-1].replace(" ", ""))
        self.c.execute("unmark", str(coords)[1:-1].replace(" ", ""))

    def turn_and_go(self, direction):
        if direction == "south":
            left, right, back = "east", "west", "north"
        elif direction == "north":
            left, right, back = "west", "east", "south"
        elif direction == "east":
            left, right, back = "north", "south", "west"
        elif direction == "west":
            left, right, back = "south", "north", "east"
        else:
            left, right, back = (None,) * 3

        self_direction = self.get_self_direction()
        if self_direction == back:
            self.c.execute("command", "right")
            self.c.execute("command", "right")
        elif self_direction == right:
            self.c.execute("command", "left")
        elif self_direction == left:
            self.c.execute("command", "right")
        self.c.execute("command", "forward")

    def follow_path(self, path):
        self.c.execute("command", "set_steps")
        for node in path:
            coords = node[0]
            position = self.current_position()
            dx, dy = coords[0]-position[0], coords[1]-position[1]

            if abs(dx) != 1:
                dx = -dx
            if abs(dy) != 1:
                dy = -dy

            if dy > 0:
                self.turn_and_go("south")  # , "east", "west", "north")
            elif dy < 0:
                self.turn_and_go("north")  # , "west", "east", "south")
            elif dx > 0:
                self.turn_and_go("east")  # , "north", "south", "west")
            else:
                self.turn_and_go("west")  # , "south", "north", "east")
        input("Waiting for return")

    def get_self_direction(self):
        return self.c.execute("info", "direction")

    def opposite_direction(self, direction):
        return self.opposite_directions.get(direction, None)
