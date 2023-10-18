import random
import matplotlib.pyplot as plot


class Game:
    POSSIBLE_MOVE = ("KAMIEŃ", "PAPIER", "NOŻYCE")
    TRAINING_LOOP_COUNT = 100000
    # W macierzy prawdopodobieństw używam liczb całkowitych, aby uniknąć błędów zaokrągleń i zbędnej normalizacji
    probability_matrix = ([1, 1, 1], [1, 1, 1], [1, 1, 1])
    previous_move = None
    score_history = [0]

    def __init__(self):
        while True:
            train_before_start = input(f"Witaj! Czy chcesz nauczyć SI przed rozgrywką? (T/N): ").upper()

            if train_before_start not in ("T", "N"):
                continue

            if train_before_start == "T":
                self.train()
                self.start()
                break

            if train_before_start == "N":
                self.start()
                break

    def get_most_probable_counter_move(self):
        if self.previous_move is None:
            return random.choice(self.POSSIBLE_MOVE)

        matrix_row_by_previous_move = self.probability_matrix[self.POSSIBLE_MOVE.index(self.previous_move)]
        most_probable_move = random.choices(self.POSSIBLE_MOVE, weights=matrix_row_by_previous_move)[0]

        return self.POSSIBLE_MOVE[0] if most_probable_move == self.POSSIBLE_MOVE[2] else self.POSSIBLE_MOVE[
            self.POSSIBLE_MOVE.index(most_probable_move) + 1]

    def update_probability_matrix(self, move):
        if self.previous_move is None:
            return

        move_index = self.POSSIBLE_MOVE.index(move)
        matrix_row_by_previous_move = self.probability_matrix[self.POSSIBLE_MOVE.index(self.previous_move)]
        matrix_row_by_previous_move[move_index] = matrix_row_by_previous_move[move_index] + 1

    def get_result_score(self, user_move, counter_move):
        if user_move == counter_move:
            return 0

        return 1 if (self.POSSIBLE_MOVE.index(user_move) - self.POSSIBLE_MOVE.index(counter_move)) % 3 == 1 else -1

    def update_score_history(self, score):
        self.score_history.append(self.score_history[-1] + score)

    def show_score_history_plot(self):
        plot.plot(self.score_history)
        plot.xlabel("Numer rozgrywki")
        plot.ylabel("Wynik gracza (SI w przypadku nauki)")
        plot.show()

    def train(self):
        for _ in range(self.TRAINING_LOOP_COUNT):
            random_move = random.choice(self.POSSIBLE_MOVE)

            self.update_probability_matrix(random_move)

            counter_move = self.get_most_probable_counter_move()
            result_score = self.get_result_score(counter_move, random_move)
            self.update_score_history(result_score)

            # print(f"Losowy ruch: {random_move.capitalize()}, ruch SI: {counter_move.capitalize()}")
            # print(f"Wynik SI: {self.score_history[-1]}")

            self.previous_move = random_move

        print(f"Nauczono SI na próbie {self.TRAINING_LOOP_COUNT} pseudolosowych rozgrywek!")
        print(f"Macierz prawdopodobieństw: {self.probability_matrix}")

        self.previous_move = None
        self.show_score_history_plot()
        self.score_history = [0]

    def start(self):
        print("Zakończenie gry - STOP")

        while True:
            user_move = input(f"Twój ruch ({', '.join(move.capitalize() for move in self.POSSIBLE_MOVE)}): ").upper()

            if user_move == 'STOP':
                self.show_score_history_plot()
                break

            if user_move not in self.POSSIBLE_MOVE:
                continue

            self.update_probability_matrix(user_move)

            counter_move = self.get_most_probable_counter_move()
            result_score = self.get_result_score(user_move, counter_move)
            self.update_score_history(result_score)

            print(f"Ruch SI: {counter_move.capitalize()}")
            print(f"Twój wynik: {self.score_history[-1]}")

            self.previous_move = user_move


Game()
