#!/bin/bash
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
BASEDIR=$(dirname "$SCRIPT")
# Navigate to the base directory to ensure all paths are relative to this script
cd "$BASEDIR" || { echo "Failed to navigate to the script's base directory"; exit 1; }

# Array of task names
TASKS=("niah" "variable_tracking" "qa")

# Loop through each task and execute the scripts
for TASK in "${TASKS[@]}"; do
    echo "Starting $TASK Task..."
    
    # Change to the task directory in data
    if cd "data/$TASK"; then
        # Run the task specific script
        echo "Running $TASK Task..."
        if ! python "${TASK}.py"; then
            echo "Failed to run $TASK script"
        fi
    else
        echo "Failed to change directory to data/$TASK"
    fi

    # Reset to base directory to ensure the correct path for eval
    cd "$BASEDIR"
    echo "base directory is $BASEDIR" 
    # Change directory to eval to run metrics
    if cd "eval"; then
        # Run the metrics generation script for the current task
        echo "Generating Metrics for $TASK..."
        if ! python generate.py task=$TASK; then
            echo "Failed to generate metrics for $TASK"
        fi
    else
        echo "Failed to change directory to eval"
    fi

    # Reset to base directory for the next iteration
    cd "$BASEDIR"
    echo "Current directory: $(pwd)" 
    
    echo "$TASK Task completed."
    echo "-----------------------------"
done

echo "All tasks have been completed."
