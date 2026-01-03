"""This script demonstrates a full SML interface for the Supplier Sorting agent using event-handlers."""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import Course01_SoarEssentials.run_soar          # Importing this script sets up the env variables needed for us to import Python_sml_ClientInterface.

# Define the path for our Supplier Sorting Soar agent rules (in an OS-independent way).
AGENT_PATH = str(pathlib.Path(__file__).parent.joinpath("agent_code", "_firstload_suppliersort.soar").resolve())

#########################################################################
# Main Code:


# Import the SML library
import Python_sml_ClientInterface as sml

# *After* setting up the env vars, import our helper classes and callback functions
from suppliersort_course.Project10_SML.app4.instructor_solution_io import MyAgentInputManager
from suppliersort_course.Project10_SML.app4.instructor_solution_callbacks import (
    callback_print, callback_run, callback_init_agent, 
    callback_output_handler, callback_output_notification_event, callback_output_kernel_update_event
)

# Create the Soar agent
kernel = sml.Kernel.CreateKernelInNewThread(12121)
agent = kernel.CreateAgent("My Agent")

input_manager = MyAgentInputManager(agent)

## Set up non-output event handlers

# STEP 1: Register a callback for print events
# --> Use agent.RegisterForPrintEvent(sml_event, callback, user_data)
# (In this case, we don't need to pass along any custom data, so you can pass None for user_data.)
# Don't forget to save the returned callback ID so we can unregister the callback later.
print_event_callback_id = agent.RegisterForPrintEvent(sml.smlEVENT_PRINT, callback_print, None)

# STEP 3: Register a callback for two run events: BEFORE_RUN_STARTS and BEFORE_INPUT_PHASE
# --> Use agent.RegisterForRunEvent(sml_event, callback, user_data)
#     (Call it twice, once for each event.)
# In both cases, you'll want to pass along input_manager as the user_data argument.
run_before_run_callback_id = agent.RegisterForRunEvent(sml.smlEVENT_BEFORE_RUN_STARTS, callback_run, input_manager)
run_before_input_callback_id = agent.RegisterForRunEvent(sml.smlEVENT_BEFORE_INPUT_PHASE, callback_run, input_manager)

# STEP 5: Register a callback for the agent event: BEFORE_AGENT_REINITIALIZED
# --> Use kernel.RegisterForAgentEvent(sml_event, callback, user_data)
# Again, pass along input_manager as the user_data argument.
init_agent_callback_id = kernel.RegisterForAgentEvent(sml.smlEVENT_BEFORE_AGENT_REINITIALIZED, callback_init_agent, input_manager)


## Set up the output event handler

# STEP 7:
## Callback Method 1: Listen for specific attributes on the output-link
#   Add an output handler to listen for the 'supplier-list' attribute
#   --> Use agent.AddOutputHandler(attribute_name, callback, user_data)
output_handler_callback_id = agent.AddOutputHandler("supplier-list", callback_output_handler, None)

# STEP 9:
## Callback Method 2: Listen for any change to WMEs on the output-link
#   Register a callback for output notifications
#   --> Use agent.SetOutputLinkChangeTracking(True) to enable this feature.
#   --> Then use agent.RegisterForOutputNotification(callback, user_data) to register the callback.
# agent.SetOutputLinkChangeTracking(True)
# output_notification_callback_id = agent.RegisterForOutputNotification(callback_output_notification_event, None)

## Note: Only set output-link change tracking to True if using RegisterForOutputNotification().
##   (It is unnecessary otherwise, and is more computationally expensive.)

# STEP 11:
## Callback Method 3: Check the output-link after kernel-level updates and inspect for changes manually
#   Register a callback for the kernel-level update event: AFTER_ALL_GENERATED_OUTPUT
#   --> Use kernel.RegisterForUpdateEvent(sml_event, callback, user_data).
#   Pass the agent's name ("My Agent") as the user_data argument to the callback.
# output_update_callback_id = kernel.RegisterForUpdateEvent(sml.smlEVENT_AFTER_ALL_GENERATED_OUTPUT, callback_output_kernel_update_event, "My Agent")


# Load agent productions
print(f"Loading agent files from: {AGENT_PATH}")
agent.LoadProductions(AGENT_PATH)

# STEP __: Overwrite the agent's output rule so that it doesn't create an interrupt
# --> Use agent.ExecuteCommandLine() to send an sp command to Soar.
# Name the rule "apply*suppliersort-main*output-supplier-list" so that it overwrites the existing rule.
agent.ExecuteCommandLine("""sp {apply*suppliersort-main*output-supplier-list
    "Output the supplier list without pausing the agent"
    (state <s> ^operator.name output-supplier-list
        ^supplier-list <sup-list>
        ^io.output-link <ol>)
    -->
    (<ol> ^supplier-list <sup-list>)
    (<s> ^supplier-list <sup-list> -)
	(write |*** DONE ***| (crlf) (crlf))}""")


# Run the agent 3 times
for i in range(3):
    # Run the agent once
    # (Our pre-run callback will generate supplier input before each run)
    print("\nRunning Soar!")
    agent.RunSelfForever()    # Runs the agent until our output listener stops it
    # STEP __: Remove the following line, as the agent no longer interrupts before it sends output.
    agent.RunSelf(1)
    
    # STEP __: Reset the agent's WM
    # --> Use agent.InitSoar() to init the agent.
    # (This also triggers our callback that clears candidate-supplier input)
    agent.InitSoar()


# Close resources
# (In this case, we don't actually need to unregister our callbacks,
#  since we're about to shutdown the kernel anyway,
#  but we'll do so here anyway for demonstration purposes.)

# STEP __: Unregister the print event callback
# (The others are written for you.)
if print_event_callback_id != -1:
    agent.UnregisterForPrintEvent(print_event_callback_id)
    print_event_callback_id = -1

if run_before_run_callback_id != -1:
    agent.UnregisterForRunEvent(run_before_run_callback_id)
    run_before_run_callback_id = -1

if run_before_input_callback_id != -1:
    agent.UnregisterForRunEvent(run_before_input_callback_id)
    run_before_input_callback_id = -1

if init_agent_callback_id != -1:
    kernel.UnregisterForAgentEvent(init_agent_callback_id)
    init_agent_callback_id = -1

try:
    if output_handler_callback_id != -1:
        agent.RemoveOutputHandler(output_handler_callback_id)
        output_handler_callback_id = -1
except NameError:
    pass

try:
    if output_notification_callback_id != -1:
        agent.UnregisterForOutputNotification(output_notification_callback_id)
        output_notification_callback_id = -1
except NameError:
    pass

try:
    if output_update_callback_id != -1:
        kernel.UnregisterForUpdateEvent(output_update_callback_id)
        output_update_callback_id = -1
except NameError:
    pass


print("\nGoodnight Soar.")
kernel.Shutdown()
