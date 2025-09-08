#!/bin/bash
# Tmux session name
SESSION_NAME="robot_scripts"

# Array of Python scripts to launch
# Replace these with your actual script paths
SCRIPTS=(
    "TurtleController.py"
    "TurtleSensors.py"
    "TurtleRobot_tetherless.py"
    "Cam.py"
    "Sub_Cam_ut.py"
    "logger.py"
    "untethered_planning_node.py"
)

# Function to display usage
usage() {
echo "Usage: $0 [--attach] [--kill] [--terminate] [--detach] [--live] [--logging]"
echo " --attach: Attach to the existing tmux session"
echo " --kill: Kill the existing tmux session"
echo " --terminate: Gracefully terminate Python scripts"
echo " --detach: Detach from the current tmux session"
echo " --live: Create session with live output (no log files)"
echo " --logging: Create session with output logged to files"
echo " (default): Create session with logging enabled"
exit 1
}

# Gracefully terminate Python scripts in the tmux session
terminate_scripts() {
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
# Send Ctrl-C to each pane
num_scripts=${#SCRIPTS[@]}
for ((i=0; i<num_scripts; i++)); do
tmux send-keys -t $SESSION_NAME:0.$i C-c
done
echo "Sent termination signal to all Python scripts"
# Optional: Wait a moment to allow scripts to clean up
sleep 2
else
echo "No active tmux session found"
fi
}

# Kill existing session if it exists
kill_session() {
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
tmux kill-session -t $SESSION_NAME
echo "Killed tmux session: $SESSION_NAME"
fi
}

# Detach from the current tmux session
detach_session() {
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
tmux detach-client -s $SESSION_NAME
echo "Detached from tmux session: $SESSION_NAME"
else
echo "No active tmux session found"
fi
}

# Create logs directory with timestamp
create_log_dir() {
if [ "$LOGGING_ENABLED" = true ]; then
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Created log directory: $LOG_DIR"
fi
}

# Create and setup tmux session
create_session() {
num_scripts=${#SCRIPTS[@]}
total_panes=$((num_scripts + 1)) # Scripts + 1 extra pane

# Create logs directory if logging is enabled
create_log_dir

# Create new session
tmux new-session -d -s $SESSION_NAME

# Create additional panes (total_panes - 1, since we start with 1 pane)
for ((i=1; i<total_panes; i++)); do
if [ $i -eq 1 ]; then
# First split - horizontal
tmux split-window -h
elif [ $((i % 2)) -eq 0 ]; then
# Even numbered panes - split vertically on left side
tmux select-pane -t 0
tmux split-window -v
else
# Odd numbered panes - split vertically on right side
tmux select-pane -t $((i-1))
tmux split-window -v
fi
done

# Launch scripts in the first num_scripts panes (leave the last pane free)
for ((i=0; i<num_scripts; i++)); do
if tmux select-pane -t $i 2>/dev/null; then
script_name=$(basename "${SCRIPTS[i]}" .py)
if [ "$LOGGING_ENABLED" = true ]; then
# Logging mode: pipe to both screen and log file
log_file="$LOG_DIR/${script_name}.log"
tmux send-keys "python3 ${SCRIPTS[i]} 2>&1 | tee $log_file" C-m
echo "Started ${SCRIPTS[i]} with logging to $log_file"
else
# Live mode: direct output only
tmux send-keys "python3 ${SCRIPTS[i]}" C-m
echo "Started ${SCRIPTS[i]} with live output"
fi
else
echo "Warning: Could not select pane $i"
fi
done

# Select the extra pane (last one) so user can type commands
tmux select-pane -t $num_scripts

# Show session info
if [ "$LOGGING_ENABLED" = true ]; then
echo "All scripts started with logging enabled in: $LOG_DIR"
else
echo "All scripts started with live output (no logging)"
fi
}

# Default to logging enabled (opposite of live)
LOGGING_ENABLED=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
case $1 in
--live)
LOGGING_ENABLED=false
shift
;;
--logging)
LOGGING_ENABLED=true
shift
;;
--attach)
tmux attach -t $SESSION_NAME
exit 0
;;
--kill)
kill_session
exit 0
;;
--terminate)
terminate_scripts
exit 0
;;
--detach)
detach_session
exit 0
;;
"")
# Default case - will be handled below
break
;;
*)
usage
;;
esac
done

# If no specific action was requested, create a new session
kill_session # Ensure no existing session interferes
create_session

if [ "$LOGGING_ENABLED" = true ]; then
echo "Created tmux session with logging: $SESSION_NAME"
else
echo "Created tmux session with live output: $SESSION_NAME"
fi

# Attach to the session (optional - remove if you don't want auto-attach)
tmux attach -t $SESSION_NAME

exit 0