#!/bin/bash
# Tmux session name
SESSION_NAME="robot_scripts"
# Array of Python scripts to launch
# Replace these with your actual script paths
SCRIPTS=(
"turtle_tracker.py"
"planning_node.py"
"TurtleController.py"
"controller_logger.py"
)

# Function to display usage
usage() {
    echo "Usage: $0 [--attach] [--kill] [--terminate] [--detach] [--live]"
    echo " --attach: Attach to the existing tmux session"
    echo " --kill: Kill the existing tmux session"
    echo " --terminate: Gracefully terminate Python scripts"
    echo " --detach: Detach from the current tmux session"
    echo " --live: Create session with live output (no log files)"
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
    num_scripts=${#SCRIPTS[@]}
    total_panes=$((num_scripts + 1))  # Scripts + 1 extra pane
    
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
            if [ "$LIVE_OUTPUT" = true ]; then
                tmux send-keys "python3 ${SCRIPTS[i]}" C-m
            else
                tmux send-keys "python3 ${SCRIPTS[i]} 2>&1 | tee script_${i}_log.txt" C-m
            fi
        else
            echo "Warning: Could not select pane $i"
        fi
    done
    
    # Select the extra pane (last one) so user can type commands
    tmux select-pane -t $num_scripts
}

# Add a flag for live output
LIVE_OUTPUT=false

# Check for --live flag
if [[ "$1" == "--live" ]]; then
    LIVE_OUTPUT=true
    shift # Remove --live from arguments
fi

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
    --live)
        LIVE_OUTPUT=true
        kill_session # Ensure no existing session interferes
        create_session
        echo "Created tmux session with live output: $SESSION_NAME"
        tmux attach -t $SESSION_NAME
        ;;
    "")
        kill_session # Ensure no existing session interferes
        create_session
        echo "Created tmux session: $SESSION_NAME"
        ;;
    *)
        usage
        ;;
esac

exit 0