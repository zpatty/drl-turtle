#!/bin/bash

# Tmux session name
SESSION_NAME="robot_scripts"

# Array of Python scripts to launch
# Replace these with your actual script paths
SCRIPTS=(
    "TurtleController.py"
    "TurtleRobot.py"
    "Cam.py"
    "Sub_Cam_ut.py"
    "logger.py"
    "TurtleSensors.py"
    "untethered_planning_node.py"
)
# Function to display usage
usage() {
    echo "Usage: $0 [--attach] [--kill] [--terminate] [--detach]"
    echo "  --attach: Attach to the existing tmux session"
    echo "  --kill: Kill the existing tmux session"
    echo "  --terminate: Gracefully terminate Python scripts"
    echo "  --detach: Detach from the current tmux session"
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

# Create and setup tmux session
create_session() {
    # Create new session
    tmux new-session -d -s $SESSION_NAME

    # Setup panes for 7 scripts
    # First, create 3 panes on the left
    tmux split-window -h    # First horizontal split
    tmux select-pane -t 0
    tmux split-window -v    # First vertical split on left side

    # Create 4 panes on the right
    tmux select-pane -t 2
    tmux split-window -v    # First vertical split on right
    tmux select-pane -t 1
    tmux split-window -v    # First vertical split on left
    tmux select-pane -t 3
    tmux split-window -v    # First vertical split on left middle
    tmux select-pane -t 5
    tmux split-window -v    # First vertical split on right middle

    # Careful script launching
    num_scripts=${#SCRIPTS[@]}
    for ((i=0; i<num_scripts; i++)); do
        # Use a try-catch approach to handle potential pane errors
        if tmux select-pane -t $i 2>/dev/null; then
            tmux send-keys "python3 ${SCRIPTS[i]} > script_${i}_log.txt 2>&1" C-m
        else
            echo "Warning: Could not select pane $i"
        fi
    done
}

# Main script logic
case "$1" in
    --attach)
        tmux attach -t $SESSION_NAME
        ;;
    --kill)
        kill_session
        ;;
    --terminate)
        terminate_scripts
        ;;
    --detach)
        detach_session
        ;;
    "")
        kill_session  # Ensure no existing session interferes
        create_session
        echo "Created tmux session: $SESSION_NAME"
        ;;
    *)
        usage
        ;;
esac

exit 0