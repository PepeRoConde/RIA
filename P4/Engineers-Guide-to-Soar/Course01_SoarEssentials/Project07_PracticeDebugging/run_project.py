import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from Course01_SoarEssentials.run_soar import run_soar_debugger

# Pick which file to run from this project folder
agent_file = "./agent_starter.soar"
# agent_file = "./instructor_solution.soar"

# Star Soar with a debugger using the indicated agent file
run_soar_debugger(agent_file)
