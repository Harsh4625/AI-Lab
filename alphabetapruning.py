# Alpha-Beta Pruning Complete Implementation for Google Colab
# Install required packages (run this cell first in Colab)
!pip install matplotlib pandas numpy

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

# ==================================================================================
# PART 1: BASIC ALPHA-BETA PRUNING WITH GAME TREE
# ==================================================================================

class GameTreeNode:
    """Represents a node in the game tree"""
    def __init__(self, value: Optional[int] = None, children: List['GameTreeNode'] = None, name: str = ""):
        self.value = value
        self.children = children or []
        self.name = name
        self.pruned = False
        self.alpha_at_node = float('-inf')
        self.beta_at_node = float('inf')

class AlphaBetaPruning:
    """Complete Alpha-Beta Pruning implementation with visualization"""
    
    def __init__(self):
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        self.evaluation_path = []
        self.tree_visualization = []
        
    def alpha_beta_search(self, node: GameTreeNode, depth: int, alpha: float, beta: float, 
                         maximizing_player: bool, max_depth: int = 4) -> Tuple[int, str]:
        """Enhanced alpha-beta pruning with detailed tracking"""
        self.nodes_evaluated += 1
        node.alpha_at_node = alpha
        node.beta_at_node = beta
        
        # Store evaluation step
        player_type = "MAX" if maximizing_player else "MIN"
        self.evaluation_path.append(f"Evaluating {node.name} ({player_type}): α={alpha:.1f}, β={beta:.1f}")
        
        # Base case: leaf node or max depth
        if depth >= max_depth or not node.children:
            self.evaluation_path.append(f"Leaf {node.name}: returning value {node.value}")
            return node.value, node.name
        
        best_path = ""
        
        if maximizing_player:
            max_eval = float('-inf')
            for i, child in enumerate(node.children):
                if child.pruned:
                    continue
                    
                eval_score, child_path = self.alpha_beta_search(
                    child, depth + 1, alpha, beta, False, max_depth
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_path = f"{node.name} -> {child_path}"
                    
                alpha = max(alpha, eval_score)
                self.evaluation_path.append(f"MAX {node.name}: updated α = {alpha}")
                
                # Pruning condition
                if beta <= alpha:
                    self.evaluation_path.append(f"🔥 PRUNING at {node.name}: β({beta}) ≤ α({alpha})")
                    self.pruned_branches += 1
                    # Mark remaining children as pruned
                    for j in range(i + 1, len(node.children)):
                        node.children[j].pruned = True
                    break
                    
            return max_eval, best_path
            
        else:  # Minimizing player
            min_eval = float('inf')
            for i, child in enumerate(node.children):
                if child.pruned:
                    continue
                    
                eval_score, child_path = self.alpha_beta_search(
                    child, depth + 1, alpha, beta, True, max_depth
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_path = f"{node.name} -> {child_path}"
                    
                beta = min(beta, eval_score)
                self.evaluation_path.append(f"MIN {node.name}: updated β = {beta}")
                
                # Pruning condition
                if beta <= alpha:
                    self.evaluation_path.append(f"🔥 PRUNING at {node.name}: β({beta}) ≤ α({alpha})")
                    self.pruned_branches += 1
                    # Mark remaining children as pruned
                    for j in range(i + 1, len(node.children)):
                        node.children[j].pruned = True
                    break
                    
            return min_eval, best_path

def create_demo_game_tree() -> GameTreeNode:
    """Create a comprehensive demo game tree"""
    # Leaf nodes with random values
    leaves = [
        GameTreeNode(value=3, name="L1"), GameTreeNode(value=12, name="L2"),
        GameTreeNode(value=8, name="L3"), GameTreeNode(value=2, name="L4"),
        GameTreeNode(value=4, name="L5"), GameTreeNode(value=6, name="L6"),
        GameTreeNode(value=14, name="L7"), GameTreeNode(value=5, name="L8"),
        GameTreeNode(value=2, name="L9"), GameTreeNode(value=1, name="L10"),
        GameTreeNode(value=7, name="L11"), GameTreeNode(value=9, name="L12")
    ]
    
    # Internal nodes
    min_nodes = [
        GameTreeNode(children=leaves[0:3], name="B1"),
        GameTreeNode(children=leaves[3:6], name="B2"),
        GameTreeNode(children=leaves[6:9], name="B3"),
        GameTreeNode(children=leaves[9:12], name="B4")
    ]
    
    max_nodes = [
        GameTreeNode(children=min_nodes[0:2], name="A1"),
        GameTreeNode(children=min_nodes[2:4], name="A2")
    ]
    
    root = GameTreeNode(children=max_nodes, name="ROOT")
    return root

# ==================================================================================
# PART 2: TIC-TAC-TOE IMPLEMENTATION
# ==================================================================================

class TicTacToeAI:
    """Tic-Tac-Toe with Alpha-Beta Pruning AI"""
    
    def __init__(self):
        self.board = [' '] * 9
        self.human = 'O'
        self.ai = 'X'
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        
    def print_board(self):
        """Display current board state"""
        print("\n Current Board:")
        for i in range(0, 9, 3):
            print(f" {self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6:
                print("-----------")
        print()
        
    def available_moves(self) -> List[int]:
        """Get available move positions"""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def check_winner(self) -> Optional[str]:
        """Check for winner or tie"""
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for combo in wins:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                return self.board[combo[0]]
        return 'Tie' if ' ' not in self.board else None
    
    def evaluate_board(self) -> int:
        """Evaluate current board position"""
        winner = self.check_winner()
        if winner == self.ai:
            return 10
        elif winner == self.human:
            return -10
        return 0
    
    def alpha_beta_minimax(self, depth: int, alpha: float, beta: float, is_maximizing: bool) -> int:
        """Alpha-Beta Minimax for Tic-Tac-Toe"""
        self.nodes_evaluated += 1
        
        winner = self.check_winner()
        if winner is not None:
            return self.evaluate_board()
            
        if is_maximizing:
            max_eval = float('-inf')
            for move in self.available_moves():
                self.board[move] = self.ai
                eval_score = self.alpha_beta_minimax(depth + 1, alpha, beta, False)
                self.board[move] = ' '
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    self.pruned_branches += 1
                    break
                    
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.available_moves():
                self.board[move] = self.human
                eval_score = self.alpha_beta_minimax(depth + 1, alpha, beta, True)
                self.board[move] = ' '
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.pruned_branches += 1
                    break
                    
            return min_eval
    
    def get_best_move(self) -> int:
        """Find best move using Alpha-Beta Pruning"""
        best_move = -1
        best_value = float('-inf')
        self.nodes_evaluated = 0
        self.pruned_branches = 0
        
        move_analysis = []
        
        for move in self.available_moves():
            self.board[move] = self.ai
            move_value = self.alpha_beta_minimax(0, float('-inf'), float('inf'), False)
            self.board[move] = ' '
            
            move_analysis.append((move, move_value))
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        # Display analysis
        print("🤖 AI Move Analysis:")
        for move, value in move_analysis:
            status = "⭐ BEST" if move == best_move else ""
            print(f"   Position {move}: value = {value:2d} {status}")
            
        return best_move

# ==================================================================================
# PART 3: PERFORMANCE COMPARISON & ANALYSIS
# ==================================================================================

class PerformanceAnalyzer:
    """Analyze Alpha-Beta Pruning performance vs standard Minimax"""
    
    def __init__(self):
        self.results = []
    
    def minimax(self, depth: int, node_index: int, maximizing: bool, values: List[int], height: int) -> Tuple[int, int]:
        """Standard minimax implementation"""
        nodes = 1  # Count this node
        
        if depth == height:
            return values[node_index], nodes
        
        if maximizing:
            best = float('-inf')
            total_nodes = nodes
            for i in range(2):
                val, child_nodes = self.minimax(depth + 1, node_index * 2 + i, False, values, height)
                best = max(best, val)
                total_nodes += child_nodes
            return best, total_nodes
        else:
            best = float('inf')
            total_nodes = nodes
            for i in range(2):
                val, child_nodes = self.minimax(depth + 1, node_index * 2 + i, True, values, height)
                best = min(best, val)
                total_nodes += child_nodes
            return best, total_nodes
    
    def alpha_beta_minimax(self, depth: int, node_index: int, maximizing: bool, 
                          values: List[int], alpha: float, beta: float, height: int) -> Tuple[int, int, int]:
        """Alpha-beta minimax with node counting"""
        nodes = 1
        pruned = 0
        
        if depth == height:
            return values[node_index], nodes, pruned
        
        if maximizing:
            best = float('-inf')
            total_nodes = nodes
            total_pruned = pruned
            
            for i in range(2):
                val, child_nodes, child_pruned = self.alpha_beta_minimax(
                    depth + 1, node_index * 2 + i, False, values, alpha, beta, height
                )
                best = max(best, val)
                total_nodes += child_nodes
                total_pruned += child_pruned
                alpha = max(alpha, best)
                
                if beta <= alpha:
                    total_pruned += 1
                    break
                    
            return best, total_nodes, total_pruned
        else:
            best = float('inf')
            total_nodes = nodes
            total_pruned = pruned
            
            for i in range(2):
                val, child_nodes, child_pruned = self.alpha_beta_minimax(
                    depth + 1, node_index * 2 + i, True, values, alpha, beta, height
                )
                best = min(best, val)
                total_nodes += child_nodes
                total_pruned += child_pruned
                beta = min(beta, best)
                
                if beta <= alpha:
                    total_pruned += 1
                    break
                    
            return best, total_nodes, total_pruned
    
    def run_comparison(self, tree_heights: List[int]) -> pd.DataFrame:
        """Run comprehensive comparison"""
        results = []
        
        for height in tree_heights:
            num_leaves = 2 ** height
            values = [random.randint(1, 100) for _ in range(num_leaves)]
            
            # Test Minimax
            start_time = time.time()
            minimax_result, minimax_nodes = self.minimax(0, 0, True, values, height)
            minimax_time = (time.time() - start_time) * 1000
            
            # Test Alpha-Beta
            start_time = time.time()
            ab_result, ab_nodes, ab_pruned = self.alpha_beta_minimax(
                0, 0, True, values, float('-inf'), float('inf'), height
            )
            ab_time = (time.time() - start_time) * 1000
            
            # Calculate efficiency
            node_reduction = ((minimax_nodes - ab_nodes) / minimax_nodes) * 100
            time_reduction = ((minimax_time - ab_time) / minimax_time) * 100
            
            results.append({
                'Height': height,
                'Leaves': num_leaves,
                'Minimax_Nodes': minimax_nodes,
                'AlphaBeta_Nodes': ab_nodes,
                'Branches_Pruned': ab_pruned,
                'Node_Reduction_%': node_reduction,
                'Minimax_Time_ms': minimax_time,
                'AlphaBeta_Time_ms': ab_time,
                'Time_Reduction_%': time_reduction,
                'Values_Sample': str(values[:8]) + '...' if len(values) > 8 else str(values)
            })
        
        return pd.DataFrame(results)

# ==================================================================================
# PART 4: VISUALIZATION & MAIN EXECUTION
# ==================================================================================

def create_performance_chart(df: pd.DataFrame):
    """Create performance comparison charts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Node comparison
    ax1.bar(df['Height'] - 0.2, df['Minimax_Nodes'], 0.4, label='Minimax', alpha=0.7, color='red')
    ax1.bar(df['Height'] + 0.2, df['AlphaBeta_Nodes'], 0.4, label='Alpha-Beta', alpha=0.7, color='blue')
    ax1.set_xlabel('Tree Height')
    ax1.set_ylabel('Nodes Evaluated')
    ax1.set_title('Nodes Evaluated: Minimax vs Alpha-Beta')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Node reduction percentage
    ax2.bar(df['Height'], df['Node_Reduction_%'], color='green', alpha=0.7)
    ax2.set_xlabel('Tree Height')
    ax2.set_ylabel('Node Reduction %')
    ax2.set_title('Node Reduction Achieved by Alpha-Beta Pruning')
    ax2.grid(True, alpha=0.3)
    
    # Time comparison
    ax3.plot(df['Height'], df['Minimax_Time_ms'], 'ro-', label='Minimax', linewidth=2, markersize=8)
    ax3.plot(df['Height'], df['AlphaBeta_Time_ms'], 'bo-', label='Alpha-Beta', linewidth=2, markersize=8)
    ax3.set_xlabel('Tree Height')
    ax3.set_ylabel('Execution Time (ms)')
    ax3.set_title('Execution Time Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Branches pruned
    ax4.bar(df['Height'], df['Branches_Pruned'], color='orange', alpha=0.7)
    ax4.set_xlabel('Tree Height')
    ax4.set_ylabel('Branches Pruned')
    ax4.set_title('Number of Branches Pruned')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main_demonstration():
    """Main demonstration function"""
    print("=" * 90)
    print("🎯 ALPHA-BETA PRUNING COMPREHENSIVE DEMONSTRATION")
    print("=" * 90)
    
    # DEMO 1: Basic Game Tree
    print("\n" + "🌳 DEMO 1: BASIC GAME TREE ALPHA-BETA PRUNING")
    print("-" * 70)
    
    tree = create_demo_game_tree()
    ab_algo = AlphaBetaPruning()
    
    start_time = time.time()
    optimal_value, best_path = ab_algo.alpha_beta_search(tree, 0, float('-inf'), float('inf'), True, 3)
    execution_time = (time.time() - start_time) * 1000
    
    print(f"🏆 Optimal Value: {optimal_value}")
    print(f"🛤️  Best Path: {best_path}")
    print(f"📊 Nodes Evaluated: {ab_algo.nodes_evaluated}")
    print(f"✂️  Branches Pruned: {ab_algo.pruned_branches}")
    print(f"⏱️  Execution Time: {execution_time:.2f} ms")
    
    # Show evaluation steps
    print(f"\n🔍 EVALUATION STEPS (first 10):")
    for i, step in enumerate(ab_algo.evaluation_path[:10], 1):
        print(f"{i:2d}. {step}")
    if len(ab_algo.evaluation_path) > 10:
        print(f"    ... and {len(ab_algo.evaluation_path) - 10} more steps")
    
    # DEMO 2: Tic-Tac-Toe AI
    print("\n" + "🎮 DEMO 2: TIC-TAC-TOE AI WITH ALPHA-BETA PRUNING")
    print("-" * 70)
    
    game = TicTacToeAI()
    # Set up interesting game state
    game.board = ['X', 'O', ' ', 'O', 'X', ' ', ' ', ' ', 'X']
    
    print("Current game state:")
    game.print_board()
    
    print(f"Available moves: {game.available_moves()}")
    
    start_time = time.time()
    best_move = game.get_best_move()
    ai_time = (time.time() - start_time) * 1000
    
    print(f"\n🎯 AI chooses position: {best_move}")
    print(f"📊 Nodes evaluated: {game.nodes_evaluated}")
    print(f"✂️  Branches pruned: {game.pruned_branches}")
    print(f"⏱️  AI thinking time: {ai_time:.2f} ms")
    
    # Make the move and show result
    game.board[best_move] = game.ai
    print(f"\nResult after AI move:")
    game.print_board()
    
    winner = game.check_winner()
    if winner:
        print(f"🏆 Game Result: {winner} wins!")
    
    # DEMO 3: Performance Analysis
    print("\n" + "📈 DEMO 3: PERFORMANCE ANALYSIS")
    print("-" * 70)
    
    analyzer = PerformanceAnalyzer()
    tree_heights = [3, 4, 5, 6, 7]
    
    print("Running performance comparison...")
    performance_df = analyzer.run_comparison(tree_heights)
    
    print("\n📊 PERFORMANCE RESULTS:")
    print(performance_df.to_string(index=False, float_format='%.2f'))
    
    # Create visualization
    print("\n📈 Creating performance charts...")
    create_performance_chart(performance_df)
    
    # Save results
    performance_df.to_csv('alpha_beta_performance_colab.csv', index=False)
    print("\n💾 Results saved to 'alpha_beta_performance_colab.csv'")
    
    # Summary statistics
    avg_node_reduction = performance_df['Node_Reduction_%'].mean()
    max_node_reduction = performance_df['Node_Reduction_%'].max()
    avg_pruned = performance_df['Branches_Pruned'].mean()
    
    print(f"\n🎯 SUMMARY STATISTICS:")
    print(f"   Average node reduction: {avg_node_reduction:.1f}%")
    print(f"   Maximum node reduction: {max_node_reduction:.1f}%")
    print(f"   Average branches pruned: {avg_pruned:.1f}")
    
    return performance_df

# ==================================================================================
# MAIN EXECUTION - RUN THIS IN COLAB
# ==================================================================================

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run the complete demonstration
    results_df = main_demonstration()
    
    print("\n" + "=" * 90)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 90)
    print("\n🎓 KEY TAKEAWAYS:")
    print("   • Alpha-Beta pruning significantly reduces nodes evaluated")
    print("   • Performance improvement increases with tree complexity")
    print("   • Pruning effectiveness depends on move ordering")
    print("   • Real-world applications show substantial efficiency gains")
    
    # Interactive section for Colab users
    print(f"\n🚀 INTERACTIVE SECTION:")
    print("   • Modify tree structures in create_demo_game_tree()")
    print("   • Try different Tic-Tac-Toe positions")
    print("   • Experiment with different tree heights")
    print("   • Analyze the generated CSV file")
