#!/bin/bash

# Calculate sleep duration in seconds (1h30m = 90 minutes = 5400 seconds)
sleep_duration=5400

# Print start time
echo "Current time: $(date)"
echo "Script will start in 1 hour and 30 minutes at: $(date -d "+90 minutes")"

# Sleep for the specified duration
sleep $sleep_duration

# Activate conda environment if needed (uncomment and modify if you're using conda)
# source /path/to/conda/bin/activate your_env_name

# Run the Python script
echo "Starting script at: $(date)"
python claude_generator.py

# Print completion time
echo "Script completed at: $(date)"