import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from Course01_SoarEssentials.run_soar import run_soar_debugger

# Run an empty Soar agent with a Soar Debugger
run_soar_debugger()

### Troubleshooting:
# * If you see an error about the Python_sml_ClientInterface module not being found,
#   make sure your system's security settings aren't blocking the setup scripts.
#   Try running the setup script manually from the SoarSuite folder.
# * If you see a "Permission denied" error above a module not found error, you may need to
#   change the permissions of the SoarSuite folder to include read and execute permissions.
# * If you see an error about the file 'javaw' not being found, you may need to install Java.
###
