import numpy as np
import matplotlib.pyplot as plt
import time

class NQueensCSP:
    
    def __init__(self, n):
        self.n = n
        self.board = [-1] * n
        self.solutions = []
        self.backtrack_count = 0
        
    def is_safe(self, row, col):
        for prev_row in range(row):
            prev_col = self.board[prev_row]
            
            if prev_col == col:
                return False
            
            if abs(prev_row - row) == abs(prev_col - col):
                return False
        
        return True
    
    def forward_check(self, row):
        valid_cols = []
        for col in range(self.n):
            if self.is_safe(row, col):
                valid_cols.append(col)
        return valid_cols
    
    def backtrack(self, row):
        self.backtrack_count += 1
        
        if row == self.n:
            self.solutions.append(self.board.copy())
            return True
        
        valid_cols = self.forward_check(row)
        
        for col in valid_cols:
            self.board[row] = col
            self.backtrack(row + 1)
            self.board[row] = -1
        
        return False
    
    def solve(self, find_all=True):
        self.solutions = []
        self.backtrack_count = 0
        start_time = time.time()
        
        self.backtrack(0)
        
        end_time = time.time()
        
        print(f"N-Queens Problem (N={self.n})")
        print(f"Solutions found: {len(self.solutions)}")
        print(f"Backtrack calls: {self.backtrack_count}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        
        return self.solutions
    
    def visualize_solution(self, solution_index=0):
        if not self.solutions:
            print("No solutions to visualize!")
            return
        
        if solution_index >= len(self.solutions):
            print(f"Invalid solution index! Only {len(self.solutions)} solutions available.")
            return
        
        solution = self.solutions[solution_index]
        
        board_visual = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if (i + j) % 2 == 0:
                    board_visual[i][j] = 0.8
                else:
                    board_visual[i][j] = 0.3
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(board_visual, cmap='gray', vmin=0, vmax=1)
        
        for row in range(self.n):
            col = solution[row]
            ax.text(col, row, '♛', fontsize=40, ha='center', va='center', 
                   color='red', weight='bold')
        
        for i in range(self.n + 1):
            ax.axhline(i - 0.5, color='black', linewidth=2)
            ax.axvline(i - 0.5, color='black', linewidth=2)
        
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.set_xticklabels(range(self.n))
        ax.set_yticklabels(range(self.n))
        ax.set_title(f'N-Queens Solution #{solution_index + 1} (N={self.n})', 
                    fontsize=16, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def print_solution(self, solution_index=0):
        if not self.solutions:
            print("No solutions to print!")
            return
        
        solution = self.solutions[solution_index]
        print(f"\nSolution #{solution_index + 1}:")
        print(f"Queen positions (row, col): {[(i, solution[i]) for i in range(self.n)]}")
        print("\nBoard representation:")
        
        for row in range(self.n):
            line = ""
            for col in range(self.n):
                if solution[row] == col:
                    line += "♛ "
                else:
                    line += ". "
            print(line)


print("="*60)
print("N-QUEENS CONSTRAINT SATISFACTION PROBLEM")
print("="*60)

print("\n### Example 1: 4-Queens Problem ###\n")
solver_4 = NQueensCSP(4)
solutions_4 = solver_4.solve()
solver_4.print_solution(0)
solver_4.visualize_solution(0)

print("\n### Example 2: 8-Queens Problem (Classic) ###\n")
solver_8 = NQueensCSP(8)
solutions_8 = solver_8.solve()
solver_8.print_solution(0)
solver_8.visualize_solution(0)

if len(solutions_8) > 1:
    print("\nShowing another solution...")
    solver_8.visualize_solution(1)

print("\n### Example 3: Performance Analysis ###\n")
for n in [4, 6, 8, 10, 12]:
    solver = NQueensCSP(n)
    solver.solve()
    print()

print("\n### Example 4: Custom N-Queens ###\n")
print("You can modify the value of N below to solve different board sizes:")
N = 6
solver_custom = NQueensCSP(N)
solutions_custom = solver_custom.solve()
if solutions_custom:
    solver_custom.print_solution(0)
    solver_custom.visualize_solution(0)

***output:
Solutions found: 2

Backtrack calls: 17

Time taken: 0.0000 seconds

First solution: Queens at positions (0,1), (1,3), (2,0), (3,2)

text
. ♛ . . 
. . . ♛ 
♛ . . . 
. . ♛ . 

***
