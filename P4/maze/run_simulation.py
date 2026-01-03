#!/usr/bin/env python3
"""
Maze Navigator Simulation
Run a Soar agent navigating through a randomly generated maze.
Supports multiple navigation strategies.
"""

import sys
import time
from pathlib import Path

from enviroment.maze import Maze
from enviroment.agent_interface import MazeAgent


def run_simulation(maze_width=10, maze_height=10, wall_prob=0.25, max_steps=200, 
                   visualize=True, strategy='naive'):
    """
    Run the maze navigation simulation.
    
    Args:
        maze_width: Width of the maze
        maze_height: Height of the maze
        wall_prob: Probability of a cell being a wall (0.0 to 1.0)
        max_steps: Maximum number of steps before stopping
        visualize: Whether to display the maze after each step
        strategy: Navigation strategy ('naive' or 'dfs')
    """
    print("=" * 60)
    print("SOAR MAZE NAVIGATOR SIMULATION")
    print(f"Strategy: {strategy.upper()}")
    print("=" * 60)
    
    # Create maze
    print(f"\nGenerating {maze_width}x{maze_height} maze...")
    maze = Maze(width=maze_width, height=maze_height, wall_probability=wall_prob)
    
    # Create agent with selected strategy
    print(f"Initializing Soar agent with '{strategy}' strategy...")
    agent = MazeAgent(maze, strategy=strategy)
    
    # Display initial maze
    if visualize:
        print("\nInitial Maze:")
        orientation = agent.ORIENTATION_NAMES.get(agent.orientation) if strategy == 'dfs' else None
        maze.display(agent_pos=maze.start_pos, orientation=orientation)
    
    if not agent.initialize_soar():
        print("ERROR: Failed to initialize Soar agent")
        return False
    
    print("Agent initialized successfully!")
    print(f"Start: {maze.start_pos}, Target: {maze.target_pos}")
    print("\nStarting navigation...")
    print("-" * 60)
    
    # Run simulation
    step = 0
    start_time = time.time()
    
    try:
        while step < max_steps:
            step += 1
            
            # Run one decision cycle
            agent.run_step()
            
            # Visualize if requested
            if visualize and step % 5 == 0:  # Show every 5 steps
                orientation = agent.ORIENTATION_NAMES.get(agent.orientation) if strategy == 'dfs' else None
                maze.display(agent_pos=(agent.x, agent.y), orientation=orientation)
                time.sleep(0.1)  # Small delay for visualization
            
            # Check if target reached
            if agent.has_reached_target():
                print("\n" + "=" * 60)
                print("SUCCESS! Agent reached the target!")
                print("=" * 60)
                break
        
        else:
            print("\n" + "=" * 60)
            print(f"Simulation stopped after {max_steps} steps")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Display final state
        elapsed_time = time.time() - start_time
        
        if visualize:
            print("\nFinal Maze State:")
            orientation = agent.ORIENTATION_NAMES.get(agent.orientation) if strategy == 'dfs' else None
            maze.display(agent_pos=(agent.x, agent.y), orientation=orientation)
        
        # Print statistics
        stats = agent.get_stats()
        print("\nSimulation Statistics:")
        print(f"  Strategy: {strategy}")
        print(f"  Steps taken: {stats['steps']}")
        print(f"  Cells explored: {stats['discovered']}/{stats['total_cells']}")
        print(f"  Final position: {stats['position']}")
        if stats.get('orientation'):
            print(f"  Final orientation: {stats['orientation']}")
        print(f"  Target position: {stats['target']}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        
        # Display path
        print(f"\nPath taken ({len(agent.path)} positions):")
        for i, pos in enumerate(agent.path[:20]):  # Show first 20 positions
            print(f"  {i}: {pos}")
        if len(agent.path) > 20:
            print(f"  ... ({len(agent.path) - 20} more positions)")
        
        # Clean shutdown
        print("\nShutting down Soar...")
        agent.shutdown()
        print("Simulation complete!")
    
    return agent.has_reached_target()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Soar Maze Navigator simulation")
    parser.add_argument("--width", type=int, default=10, help="Maze width (default: 10)")
    parser.add_argument("--height", type=int, default=10, help="Maze height (default: 10)")
    parser.add_argument("--walls", type=float, default=0.25, help="Wall probability 0.0-1.0 (default: 0.25)")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps (default: 200)")
    parser.add_argument("--no-visual", action="store_true", help="Disable visualization")
    parser.add_argument("--strategy", type=str, default="dfs", 
                       choices=["naive", "dfs"],
                       help="Navigation strategy: 'naive' (target-seeking) or 'dfs' (left-hand wall following)")
    
    args = parser.parse_args()
    
    success = run_simulation(
        maze_width=args.width,
        maze_height=args.height,
        wall_prob=args.walls,
        max_steps=args.max_steps,
        visualize=not args.no_visual,
        strategy=args.strategy
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
