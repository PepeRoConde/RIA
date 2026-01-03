# Test Suite

This directory contains tests for the Soar Maze Navigator project.

## Test Files

### Core Functionality Tests

- **test_basic.py** - Complete integration test of maze simulation with DFS agent
  - Creates a 5x5 maze
  - Initializes Soar agent
  - Runs 10 steps
  - Displays final state and statistics

- **test_soar.py** - Step-by-step Soar initialization test
  - Tests kernel creation
  - Tests agent creation
  - Tests production loading
  - Tests input link access
  - Tests running decision cycles

- **test_sensors.py** - Sensor functionality test
  - Tests sensor readings at various positions
  - Tests boundary detection
  - Tests wall detection

- **test_destroy_wme.py** - Tests the DestroyWME fix
  - Tests WME creation and destruction cycles
  - Tests input link updates
  - Tests command processing
  - Verifies no memory leaks or errors

- **test_edge_walls.py** - Tests edge wall generation
  - Compares old L-shaped path vs new diagonal staircase path
  - Tests wall generation at maze edges
  - Verifies the `_ensure_path()` fix

### Visualization Tools

- **visualize_path_difference.py** - Visual comparison of path algorithms
  - Shows side-by-side comparison of old vs new `_ensure_path()` algorithms
  - Displays maze with cleared paths highlighted
  - Shows edge wall counts before and after fix

## Running Tests

All tests require a working Soar installation. Run tests from the project root:

```bash
# Run a specific test
python tests/test_basic.py

# Run sensor test
python tests/test_sensors.py

# Run DestroyWME fix test
python tests/test_destroy_wme.py

# Visualize path algorithm comparison
python tests/visualize_path_difference.py
```

## Test Requirements

- Python 3.7+
- Soar Suite 9.6.2+ installed
- Correct Soar path configured in test files
