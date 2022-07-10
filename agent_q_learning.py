from importlib.resources import path
from re import T
import abs_agent
from redsages_15puzzle.puzzle15 import PuzzleBoard
import numpy as np
import collections
import msvcrt
import pathlib
from datetime import datetime
import json
import pickle
import os


class Agent_Q_Learning(abs_agent.AbsAgent):
    # This class provides a simple implementation of Q Learning

    def __init__(self):
        self.puzzle = PuzzleBoard.get_random_board()
        self.state_dict = {}
        self.q = np.zeros((1, 4))  # Initalize the q table with a single row
        self.state_visits = np.zeros((1, 1))
        self.convergence = []

        # Set some default values
        self.print_steps = False
        self.max_episodes = 2000
        self.max_training_visits = 1e8
        self.k_alpha = 1e8
        self.k_epsilon = 1e8
        self.gamma = .95
        self.epsilon_init = 0.2
        self.alpha_init = .1

    def train(self):
        # Multi phase training

        episodes = 1
        max_episodes = self.max_episodes

        while episodes <= max_episodes:

            action = self.epsilon_action()
            self.take_action_and_update_q(action)

            if self.puzzle.is_complete:
                self.convergence.append([episodes, self.puzzle.score])
                self.puzzle = PuzzleBoard.get_random_board()
                episodes += 1

    def take_action_and_update_q(self, action):
        # Helper to perform q update

        state_id = self.state_id
        k = self.k_alpha
        alpha = (k / (k + self.state_visits[state_id])) * self.alpha_init
        gamma = self.gamma

        if self.print_steps:
            print(self.puzzle.board, end='')
            print(f' {self.q[state_id]}', end='')
            print(f' action: {action}')

        old_score = self.puzzle.score
        self.puzzle.move_direction(action)
        new_score = self.puzzle.score
        new_state_id = self.state_id
        reward = new_score - old_score
        q_new_max = np.nanmax(self.q[new_state_id, :])

        if self.state_visits[state_id] < self.max_training_visits:
            self.q[state_id, action] = (
                                        self.q[state_id, action]
                                        + (alpha)
                                        * (reward
                                           + gamma*q_new_max
                                           - self.q[state_id, action]
                                           )
                                        )

        if self.print_steps:
            print(self.puzzle.board, end='')
            print(f' {self.q[state_id]}', end='')
            print(f' {state_id} --> {new_state_id} score: {self.puzzle.score}')

    def epsilon_action(self):
        # Pick an action using epsilon-greedy

        state_id = self.state_id
        actions = self.q[state_id]
        k = self.k_epsilon

        if self.state_visits[state_id] > self.max_training_visits:
            action = np.nanargmax(self.q[state_id])
        else:
            epsilon = (k/(k + self.state_visits[state_id]))*self.epsilon_init
            random_value = np.random.rand()
            if random_value < epsilon:
                if self.print_steps:
                    print('Randome Action')
                valid_actions = np.argwhere(~np.isnan(actions)).flatten()
                np.random.shuffle(valid_actions)
                action = valid_actions[0]
            else:
                action = np.nanargmax(actions)

        return action

    @property
    def state_id(self):

        board = self.puzzle.board

        if self.puzzle.get_tile_loc(board, 1) != (0, 0):
            # Phase 1
            loc1 = self.puzzle.get_tile_loc(board, 1)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc1, *loc16, 1)

        elif self.puzzle.get_tile_loc(board, 2) != (0, 1):
            # Phase 2

            loc2 = self.puzzle.get_tile_loc(board, 2)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc2, *loc16, 2)

        elif self.puzzle.get_tile_loc(board, 3) != (0, 2) or self.puzzle.get_tile_loc(board, 4) != (0, 3):
            # Phase 3

            loc3 = self.puzzle.get_tile_loc(board, 3)
            loc4 = self.puzzle.get_tile_loc(board, 4)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc3, *loc4, *loc16, 3)

        elif self.puzzle.get_tile_loc(board, 5) != (1, 0):
            # Phase 4

            loc5 = self.puzzle.get_tile_loc(board, 5)
            loc16 = self.puzzle.get_tile_locboard, (16)
            state_tuple = (*loc5, *loc16, 4)

        elif self.puzzle.get_tile_loc(board, 6) != (1, 1):
            # Phase 5

            loc6 = self.puzzle.get_tile_loc(board, 6)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc6, *loc16, 5)

        elif self.puzzle.get_tile_loc(board, 7) != (1, 2) or self.puzzle.get_tile_loc(board, 8) != (1, 3):
            # Phase 6

            loc7 = self.puzzle.get_tile_loc(board, 7)
            loc8 = self.puzzle.get_tile_loc(board, 8)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc7, *loc8, *loc16, 6)

        elif self.puzzle.get_tile_loc(board, 9) != (2, 0) or self.puzzle.get_tile_loc(board, 13) != (3, 0):
            # Phase 7

            loc9 = self.puzzle.get_tile_loc(board, 9)
            loc13 = self.puzzle.get_tile_locboard, (13)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc9, *loc13, *loc16, 7)

        elif self.puzzle.get_tile_loc(board, 10) != (board, 2, 1) or self.puzzle.get_tile_loc(board, 14) != (3, 1):
            # Phase 8

            loc10 = self.puzzle.get_tile_loc(board, 10)
            loc14 = self.puzzle.get_tile_loc(board, 14)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc10, *loc14, *loc16, 8)

        else:
            # Phase 9

            loc11 = self.puzzle.get_tile_loc(board, 11)
            loc12 = self.puzzle.get_tile_loc(board, 12)
            loc15 = self.puzzle.get_tile_loc(board, 15)
            loc16 = self.puzzle.get_tile_loc(board, 16)
            state_tuple = (*loc11, *loc12, *loc15, *loc16, 9)

        if state_tuple not in self.state_dict:
            # If the current state does not exist initalize it with random
            # values.

            if len(self.state_dict) == 0:
                self.state_dict[state_tuple] = 0
                self.q[0, :] = np.random.random_sample((1, self.q.shape[1]))-2

                # Remove invalid directions
                valid_dir = self.puzzle.valid_move_directions
                invalid_dir = [d for d in range(0, 4) if d not in valid_dir]
                for dir in invalid_dir:
                    self.q[0, dir] = np.nan

                self.state_visits[0] = 0

            else:
                self.q = np.append(self.q,
                                   (np.random.random_sample((1, self.q.shape[1]))-2),
                                   axis=0
                                   )

                # Remove invalid directions
                valid_dir = self.puzzle.valid_move_directions
                invalid_dir = [d for d in range(0, 4) if d not in valid_dir]
                for dir in invalid_dir:
                    self.q[-1, dir] = np.nan

                new_value = self.q.shape[0] - 1
                self.state_dict[state_tuple] = new_value
                self.state_visits = np.append(self.state_visits, 0)

        state_id = self.state_dict[state_tuple]
        self.state_visits[state_id] += 1

        return state_id

    def record_training_example(self):
        # This method allos the use to play the game and record their moves 
        # for later playback during training.

        puzzle = PuzzleBoard.get_random_board()
        init_state = np.copy(puzzle.board)
        get_direction_from_key = {
                                   b'H': 0,
                                   b'M': 1,
                                   b'P': 2,
                                   b'K': 3,
                                   }

        move_log = []

        while not puzzle.is_complete:

            os.system('cls')
            print(puzzle.board)
            print('Use Arrow Keys To Move')

            key = msvcrt.getch()
            if key == b'\x00':
                key = msvcrt.getch()
            else:
                key = None

            move_direction = get_direction_from_key[key]
            move_log.append(move_direction)
            puzzle.move_direction(move_direction)

        print(puzzle.score)

        unique_id = datetime.now().strftime('%m%d%Y%H%M%S')
        moves_file = pathlib.Path(unique_id + '.moves.json')
        init_state_file = pathlib.Path(unique_id + '.init.json')
        moves_path = pathlib.Path().cwd() / 'training_examples' / moves_file
        init_state_path = pathlib.Path().cwd() / 'training_examples' / init_state_file

        with open(moves_path, 'w', encoding='utf-8') as f:
            json.dump(move_log, f)

        with open(init_state_path, 'w', encoding='utf-8') as f:
            json.dump(init_state.tolist(), f)

    def train_on_examples(self, num_repeats, max_moves=1000):
        # Perform initial training on examples

        training_path = pathlib.Path().cwd() / 'training_examples'

        all_files = training_path.glob('*.json')
        all_ids = []
        for file in training_path.glob('*.json'):
            all_ids.append(pathlib.Path(file.stem).stem)
        
        unique_ids = np.unique(all_ids)
        unique_ids = np.tile(unique_ids, num_repeats)
        np.random.shuffle(unique_ids)
        
        training_loss = [] 

        for count, id in enumerate(unique_ids):

            init_path = pathlib.Path().cwd() / f'training_examples/{id}.init.json'
            with open(init_path) as f:
                init_array = json.load(f)
                init_array = np.array(init_array)

            moves_path = pathlib.Path().cwd() / f'training_examples/{id}.moves.json'
            with open(moves_path) as f:
                moves = json.load(f)

            # First train on example game
            try:
                self.puzzle = PuzzleBoard.get_this_board(init_array)
            except ValueError as e:
                raise TypeError('This indicates an invalid initial state'
                                'passed as training data.') from e

            for action in moves:
                self.take_action_and_update_q(action)

            try:
                self.puzzle = PuzzleBoard.get_this_board(init_array)
            except ValueError as e:
                raise TypeError('This indicates an invalid initial state'
                                'passed as training data.') from e
            
            move_count = 0
            while move_count < 2000 and not self.puzzle.is_complete:
                action = self.epsilon_action()
                self.take_action_and_update_q(action)
                move_count += 1

            agent_score = self.puzzle.score

            try:
                self.puzzle = PuzzleBoard.get_this_board(init_array)
            except ValueError as e:
                raise TypeError('This indicates an invalid initial state'
                                'passed as training data.') from e

            for action in moves:
                self.take_action_and_update_q(action)

            example_score = self.puzzle.score
            
            training_loss.append(agent_score-example_score)

        return training_loss

    def save_training(self):
        # Save the current state of the training

        q_file = pathlib.Path().cwd() / 'q.json'
        state_dict_file = pathlib.Path().cwd() / 'state_dict.pkl'

        with open(q_file, 'w', encoding='utf-8') as f:
            json.dump(self.q.tolist(), f)

        with open(state_dict_file, 'wb') as f:
            pickle.dump(self.state_dict, f)

    def load_q_table(self, q_file_input=None, state_file_input=None):
        # This method loads existing q tables and associated state dicts

        # Load default files. The files must be loaded in mathing pairs.
        if q_file_input is None and state_file_input is None:
            q_file_input = pathlib.Path().cwd / 'q.json'
            state_file_input = pathlib.Path().cwd() / 'state_dict.pkl'
        elif (q_file_input is None) ^ (state_file_input is None):

            raise ValueError('Files must be loaded in pairs. You must specify'
                             + 'both q_file_input and state_file_input'
                             + 'or neither.'
                             )

        q_path = pathlib.Path(q_file_input)
        state_path = pathlib.Path(state_file_input)

        if q_path.exists():
            with open(q_path.resolve(), 'r') as f:
                self.q = json.load(f)

        if state_path.exists():
            with open(state_path.resolve(), 'rb') as f:
                self.state_dict = pickle.load(f)

    def predict(self):
        pass
