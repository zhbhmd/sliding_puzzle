from queue import PriorityQueue


class PuzzleState:
    def __init__(self, board, g=0, parent=None):
        self.board = board
        self.size = len(board)
        self.g = g  # Cost to reach this state
        self.parent = parent
        self.goal = self.compute_goal()

    # for priorty quere sorting
    def __lt__(self, other):
        return self.g + self.h() < other.g + other.h()

    def h(self):
        """Heuristic function: sum of Manhattan distances of tiles from their goal positions."""
        return self.manhattan_distance()

    def compute_goal(self):
        goal = {}
        for i in range(self.size):
            for j in range(self.size):
                goal.update({(i * self.size + j + 1) % (self.size * self.size): (i, j)})
        return goal

    def manhattan_distance(self):
        """Calculate the sum of Manhattan distances of tiles from their goal positions."""
        distance = 0
        for r in range(self.size):
            for c in range(self.size):
                tile = self.board[r][c]
                if tile != 0:
                    goal_pos = self.goal[tile]
                    distance += abs(goal_pos[0] - r) + abs(goal_pos[1] - c)
        return distance

    def is_goal(self):
        """Check if the current state is the goal state."""
        goal = [[(i * self.size + j + 1) % (self.size * self.size) for j in range(self.size)] for i in range(self.size)]
        return self.board == goal

    def find_zero_position(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    return i, j

    def get_neighbors(self):
        """Generate neighbors of the current state."""
        neighbors = []
        x, y = self.find_zero_position()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
                neighbors.append(PuzzleState(new_board, self.g + 1, self))
        return neighbors

    def __repr__(self):
        return f"PuzzleState(board={self.board}, empty_tile_pos={self.find_zero_position()}, g={self.g})"


def is_square(state):
    rows = len(state.board)
    for i in range(rows):
        if len(state.board[i]) != rows:
            return False

    return True


def is_solvable(state):
    board = state.board
    size = len(board)
    is_even = size % 2 == 0
    inversion_count = 0
    matrix_array = []

    # convert matrix to array
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != 0:
                matrix_array.append(board[i][j])

    for i in range(len(matrix_array)):
        for j in range(i, len(matrix_array)):
            if matrix_array[i] > matrix_array[j]:
                inversion_count += 1

    if is_even:
        '''
        If even board then
        1) zero on even row -> inversion has to be even for solvability
        2) zero on odd row -> inversion has to be odd for solvability
        '''
        # zero on odd row
        if state.find_zero_position()[0] + 1 % 2 == 1:
            return inversion_count % 2 == 1, 'Solvable' if inversion_count % 2 == 1 else 'Even board, inversion count not odd, for zero in odd row'
        else:
            return inversion_count % 2 == 0, 'Solvable' if inversion_count % 2 == 0 else 'Even board, inversion count not even, for zero in even row'
    else:
        '''
        If odd board then
        1) inversion count has to be even for solvability
        '''
        return inversion_count % 2 == 0, 'Solvable' if inversion_count % 2 == 0 else 'Odd board, inversion count not even'


def a_star_search(init_state):
    max_mem = 0
    nodes_exp = 0

    if not is_square(init_state):
        return None, nodes_exp, max_mem, "Number of rows and cols are not the same"

    solvable, message = is_solvable(init_state)
    if not solvable:
        return None, nodes_exp, max_mem, message

    open_list = PriorityQueue()
    closed_set = set()
    open_list.put(init_state)

    while not open_list.empty():
        max_mem = max(max_mem, open_list.qsize() + len(closed_set))

        current_state = open_list.get()
        nodes_exp += 1

        if current_state.is_goal():
            return current_state, nodes_exp, max_mem, "Solved"

        solvable, message = is_solvable(current_state)
        if not solvable:
            return None, nodes_exp, max_mem, message

        closed_set.add(tuple(map(tuple, current_state.board)))

        for neighbor in current_state.get_neighbors():
            if tuple(map(tuple, neighbor.board)) in closed_set:
                continue
            open_list.put(neighbor)

    return None, nodes_exp, max_mem, "No solution found"


# Breath First Search BFS
def brute_force_search(init_state):
    unvisited_list = [init_state]
    visited_set = set()

    max_mem = 0
    nodes_exp = 0

    if not is_square(init_state):
        return None, nodes_exp, max_mem, "Number of rows and cols are not the same"

    solvable, message = is_solvable(init_state)
    if not solvable:
        return None, nodes_exp, max_mem, message

    while unvisited_list:
        max_mem = max(max_mem, len(unvisited_list) + len(visited_set))

        current_state = unvisited_list.pop(0)
        nodes_exp += 1

        if current_state.is_goal():
            return current_state, nodes_exp, max_mem, "Solved"

        solvable, message = is_solvable(current_state)
        if not solvable:
            return None, nodes_exp, max_mem, message

        visited_set.add(tuple(map(tuple, current_state.board)))

        for neighbor in current_state.get_neighbors():
            if tuple(map(tuple, neighbor.board)) in visited_set:
                continue
            unvisited_list.append(neighbor)  # BFS
            # unvisised_list.prepend(neighbor) # DFS

    return None, nodes_exp, max_mem, "No solution found"


def print_solution(sol):
    if sol is None:
        print("No solution found")
        return

    path = []
    state = sol
    while state:
        path.append(state.board)
        state = state.parent
    path.reverse()
    for step in path:
        for row in step:
            print(row)
        print()
    print(f"Moves: {len(path) - 1}")


def print_formatted(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            print(board[i][j], end=" ")
        print()


def run_2x2_solvable_scenario():
    initial_board = [
        [0, 2],
        [1, 3]
    ]

    print("\n2 by 2 solvable scenario: ")
    print_formatted(initial_board)
    print("Brute Force Search Solution:")
    solution_bf, nodes_expanded_bf, max_memory_bf, message = brute_force_search(PuzzleState(initial_board))
    print_solution(solution_bf)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_bf} Max Memory: {max_memory_bf}")

    print("A* Search Solution:")
    solution_astar, nodes_expanded_astar, max_memory_astar, message = a_star_search(PuzzleState(initial_board))
    print_solution(solution_astar)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_astar} Max Memory: {max_memory_astar}")


def run_2x2_not_solvable_scenario():
    initial_board = [
        [2, 1],
        [3, 0]
    ]

    print("\n2X2 not solvable scenario: ")
    print_formatted(initial_board)
    print("Brute Force Search Solution:")
    solution_bf, nodes_expanded_bf, max_memory_bf, message = brute_force_search(PuzzleState(initial_board))
    print_solution(solution_bf)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_bf} Max Memory: {max_memory_bf}")

    print("A* Search Solution:")
    solution_astar, nodes_expanded_astar, max_memory_astar, message = a_star_search(PuzzleState(initial_board))
    print_solution(solution_astar)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_astar} Max Memory: {max_memory_astar}")


def run_3x3_solvable_scenario():
    initial_board = [
        [1, 2, 3],
        [8, 5, 0],
        [4, 7, 6]
    ]

    print("\n3x3 solvable scenario: ")
    print_formatted(initial_board)
    print("Brute Force Search Solution:")
    solution_bf, nodes_expanded_bf, max_memory_bf, message = brute_force_search(PuzzleState(initial_board))
    print_solution(solution_bf)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_bf} Max Memory: {max_memory_bf}")

    print("A* Search Solution:")
    solution_astar, nodes_expanded_astar, max_memory_astar, message = a_star_search(PuzzleState(initial_board))
    print_solution(solution_astar)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_astar} Max Memory: {max_memory_astar}")


def run_3x3_not_solvable_scenario():
    initial_board = [
        [1, 2, 3],
        [8, 5, 0],
        [4, 6, 7]
    ]

    print("\n3x3 not solvable scenario: ")
    print_formatted(initial_board)
    print("Brute Force Search Solution:")
    solution_bf, nodes_expanded_bf, max_memory_bf, message = brute_force_search(PuzzleState(initial_board))
    print_solution(solution_bf)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_bf} Max Memory: {max_memory_bf}")

    print("A* Search Solution:")
    solution_astar, nodes_expanded_astar, max_memory_astar, message = a_star_search(PuzzleState(initial_board))
    print_solution(solution_astar)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_astar} Max Memory: {max_memory_astar}")


def run_new_scenario_for_presentation(initial_board):
    print("\n new scenario: ")
    print_formatted(initial_board)
    print("Brute Force Search Solution:")
    solution_bf, nodes_expanded_bf, max_memory_bf, message = brute_force_search(PuzzleState(initial_board))
    print_solution(solution_bf)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_bf} Max Memory: {max_memory_bf}")

    print("A* Search Solution:")
    solution_astar, nodes_expanded_astar, max_memory_astar, message = a_star_search(PuzzleState(initial_board))
    print_solution(solution_astar)
    print(f"Message: {message}")
    print(f"Nodes Expanded: {nodes_expanded_astar} Max Memory: {max_memory_astar}")


#run_2x2_solvable_scenario()
# run_2x2_not_solvable_scenario()
#
run_3x3_solvable_scenario()
# run_3x3_not_solvable_scenario()

new_board = [
    [1, 0, 3],
    [8, 5, 2],
    [4, 6, 7]
]

# new_board = [
#     [1, 0],
#     [2, 3]
# ]
# run_new_scenario_for_presentation(new_board)
