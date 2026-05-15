#!/bin/bash
# Autonomous Turtle Robot Launcher with Edge Impulse Tracking
SESSION_NAME="autonomous_turtle"

# ROS2 and robot nodes
NODES=(
    "python3 crush_tracker.py --model ./turtle-tracking.eim --camera-topic frames --message-type turtlecam --use-stitching --target-class turtle --confidence-threshold 0.5 --save-data"
    "python3 TurtleRobot_tetherless.py"
    "python3 Cam.py"
    "python3 TurtleSensors.py"
    "python3 TurtleController.py"
    "python3 untethered_planning_node.py"
    "python3 logger.py"
)

NODE_NAMES=(
    "EdgeImpulse_Tracker"
    "TurtleRobot"
    "Camera"
    "Sensors"
    "Controller"
    "Planner"
    "Logger"
)

# Function to display usage
usage() {
    echo "Usage: $0 [--attach] [--kill] [--terminate] [--detach] [--live] [--logging]"
    echo "  --attach:     Attach to the existing tmux session"
    echo "  --kill:       Kill the existing tmux session"
    echo "  --terminate:  Gracefully terminate all nodes (Ctrl-C)"
    echo "  --detach:     Detach from the current tmux session"
    echo "  --live:       Create session with live output (no log files)"
    echo "  --logging:    Create session with output logged to files"
    echo "  (default):    Create session with logging enabled"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start with logging"
    echo "  $0 --live           # Start without logging"
    echo "  $0 --detach         # Detach and let robot run"
    echo "  $0 --attach         # Reconnect to running session"
    echo "  $0 --terminate      # Stop all nodes gracefully"
    exit 1
}

# Gracefully terminate nodes in the tmux session
terminate_nodes() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "Sending termination signal (Ctrl-C) to all nodes..."
        num_nodes=${#NODES[@]}
        for ((i=0; i<num_nodes; i++)); do
            echo "  Stopping ${NODE_NAMES[i]}..."
            tmux send-keys -t $SESSION_NAME:0.$i C-c
        done
        echo "✓ Sent termination signals to all nodes"
        echo "  Waiting for graceful shutdown..."
        sleep 3
        echo "✓ Nodes should be stopped"
    else
        echo "✗ No active tmux session found"
    fi
}

# Kill existing session if it exists
kill_session() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux kill-session -t $SESSION_NAME
        echo "✓ Killed tmux session: $SESSION_NAME"
    else
        echo "✓ No existing session to kill"
    fi
}

# Detach from the current tmux session
detach_session() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux detach-client -s $SESSION_NAME
        echo "✓ Detached from session: $SESSION_NAME"
        echo "  Robot continues running in background"
        echo "  Reconnect with: $0 --attach"
    else
        echo "✗ No active tmux session found"
    fi
}

# Create logs directory with timestamp
create_log_dir() {
    if [ "$LOGGING_ENABLED" = true ]; then
        LOG_DIR="logs/turtle_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$LOG_DIR"
        echo "✓ Created log directory: $LOG_DIR"
    fi
}

# Create and setup tmux session
create_session() {
    num_nodes=${#NODES[@]}
    total_panes=$((num_nodes + 1))  # Nodes + 1 free command pane

    # Create logs directory if logging is enabled
    create_log_dir

    echo "Creating tmux session: $SESSION_NAME"
    echo "Nodes to launch: ${num_nodes}"
    echo ""

    # Create new session
    tmux new-session -d -s $SESSION_NAME

    # Create additional panes
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

    # Launch nodes in the first num_nodes panes
    for ((i=0; i<num_nodes; i++)); do
        if tmux select-pane -t $i 2>/dev/null; then
            node_name="${NODE_NAMES[i]}"
            
            if [ "$LOGGING_ENABLED" = true ]; then
                # Logging mode: pipe to both screen and log file
                log_file="$LOG_DIR/${node_name}.log"
                tmux send-keys "${NODES[i]} 2>&1 | tee $log_file" C-m
                echo "[$i] Started ${node_name} → $log_file"
            else
                # Live mode: direct output only
                tmux send-keys "${NODES[i]}" C-m
                echo "[$i] Started ${node_name} (live output)"
            fi
        else
            echo "✗ Warning: Could not select pane $i"
        fi
    done

    # Select the free command pane (last one)
    tmux select-pane -t $num_nodes

    echo ""
    if [ "$LOGGING_ENABLED" = true ]; then
        echo "✓ All nodes started with logging: $LOG_DIR"
    else
        echo "✓ All nodes started with live output"
    fi
    
    echo ""
    echo "Tmux Controls:"
    echo "  Ctrl-B + D     : Detach (robot keeps running)"
    echo "  Ctrl-B + Arrow : Navigate between panes"
    echo "  Ctrl-B + [     : Scroll mode (q to exit)"
    echo "  Ctrl-C in pane : Stop that specific node"
    echo ""
    echo "Script Commands:"
    echo "  $0 --detach    : Detach from session"
    echo "  $0 --attach    : Reattach to session"
    echo "  $0 --terminate : Stop all nodes gracefully"
    echo "  $0 --kill      : Force kill entire session"
}

# Default to logging enabled
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
            if tmux has-session -t $SESSION_NAME 2>/dev/null; then
                echo "✓ Attaching to session: $SESSION_NAME"
                tmux attach -t $SESSION_NAME
            else
                echo "✗ No session found. Start one with: $0"
            fi
            exit 0
            ;;
        --kill)
            kill_session
            exit 0
            ;;
        --terminate)
            terminate_nodes
            exit 0
            ;;
        --detach)
            detach_session
            exit 0
            ;;
        "")
            break
            ;;
        *)
            usage
            ;;
    esac
done

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✗ Session already exists: $SESSION_NAME"
    echo "  Options:"
    echo "    $0 --attach    # Reconnect to existing session"
    echo "    $0 --kill      # Kill existing and start fresh"
    exit 1
fi

# Create new session
echo "================================================"
echo "  Autonomous Turtle Robot Launcher"
echo "  Edge Impulse FOMO Tracking + ROS2 Control"
echo "================================================"
echo ""

create_session

# Attach to the session
echo "Attaching to session..."
sleep 1
tmux attach -t $SESSION_NAME

exit 0