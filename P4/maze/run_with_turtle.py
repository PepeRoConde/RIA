"""
Main script to run Soar maze navigation with Turtle graphics visualization.
Separates UI logic from agent logic.
"""

import time
import argparse
from enviroment.maze import Maze
from enviroment.agent_interface import MazeAgent
from enviroment.turtle_ui import MazeTurtleUI


def run_simulation(strategy: str = "dfs", width: int = 10, height: int = 10,
                   wall_prob: float = 0.3, max_steps: int = 1000,
                   delay: float = 0.1, cell_size: int = 40):
    """
    Run maze navigation simulation with Turtle visualization.

    Args:
        strategy: Navigation strategy ('naive' or 'dfs')
        width: Maze width
        height: Maze height
        wall_prob: Wall generation probability
        max_steps: Maximum steps before timeout
        delay: Delay between steps in seconds (0 for fastest)
        cell_size: Size of each cell in pixels
    """
    print("=" * 60)
    print(f"Soar Maze Navigator - {strategy.upper()} Strategy")
    print("=" * 60)

    # Create maze
    print(f"\nGenerating {width}x{height} maze (wall probability: {wall_prob})...")
    maze = Maze(width=width, height=height, wall_probability=wall_prob)

    # Initialize agent
    print(f"Initializing Soar agent with {strategy} strategy...")
    agent = MazeAgent(maze, strategy=strategy)

    if not agent.initialize_soar():
        print("Failed to initialize Soar agent!")
        return

    print("Soar agent initialized successfully!")

    # Initialize Turtle UI
    print("Initializing Turtle graphics...")
    ui = MazeTurtleUI(maze, cell_size=cell_size)

    # Set initial agent position
    ui.update_agent_position(
        agent.x, agent.y,
        agent.ORIENTATION_NAMES.get(agent.orientation) if strategy == "dfs" else None
    )
    ui.mark_discovered(agent.discovered)
    ui.update_info(agent.get_stats())

    print("\nStarting navigation...\n")

    # Run simulation
    step_count = 0
    start_time = time.time()

    try:
        while step_count < max_steps:
            # Check if target reached
            if agent.has_reached_target():
                elapsed = time.time() - start_time
                print(f"\n{'=' * 60}")
                print("TARGET REACHED!")
                print(f"{'=' * 60}")
                print(f"Steps taken: {step_count}")
                print(f"Cells discovered: {len(agent.discovered)}/{maze.width * maze.height}")
                print(f"Time elapsed: {elapsed:.2f} seconds")
                print(f"{'=' * 60}\n")

                # Show completion on UI
                ui.show_completion_message(step_count, len(agent.discovered))
                break

            # Run one step
            agent.run_step()
            step_count += 1

            # Update UI
            ui.update_agent_position(
                agent.x, agent.y,
                agent.ORIENTATION_NAMES.get(agent.orientation) if strategy == "dfs" else None
            )
            ui.mark_discovered(agent.discovered)
            ui.update_info(agent.get_stats())

            # Delay for visualization
            if delay > 0:
                time.sleep(delay)

        else:
            # Max steps reached
            print(f"\n{'=' * 60}")
            print(f"TIMEOUT: Maximum steps ({max_steps}) reached")
            print(f"{'=' * 60}")
            print(f"Final position: ({agent.x}, {agent.y})")
            print(f"Target position: {maze.target_pos}")
            print(f"Cells discovered: {len(agent.discovered)}/{maze.width * maze.height}")
            print(f"{'=' * 60}\n")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    finally:
        # Draw final path
        ui.draw_path()

        print("\nSimulation complete. Close the turtle window to exit.")

        # Shutdown Soar
        agent.shutdown()

        # Keep window open
        ui.mainloop()


def main():
    """Parse arguments and run simulation."""
    parser = argparse.ArgumentParser(
        description="Soar Maze Navigator with Turtle Graphics"
    )

    parser.add_argument(
        "--strategy",
        choices=["naive", "dfs"],
        default="dfs",
        help="Navigation strategy (default: dfs)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Maze width (default: 10)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=10,
        help="Maze height (default: 10)"
    )

    parser.add_argument(
        "--walls",
        type=float,
        default=0.3,
        help="Wall probability 0.0-1.0 (default: 0.3)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps before timeout (default: 1000)"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps in seconds (default: 0.1, use 0 for fastest)"
    )

    parser.add_argument(
        "--cell-size",
        type=int,
        default=40,
        help="Cell size in pixels (default: 40)"
    )

    args = parser.parse_args()

    run_simulation(
        strategy=args.strategy,
        width=args.width,
        height=args.height,
        wall_prob=args.walls,
        max_steps=args.max_steps,
        delay=args.delay,
        cell_size=args.cell_size
    )


if __name__ == "__main__":
    main()
