#!/bin/bash
# start.sh - main launcher

echo "Starting Flask web service..."
# Run Flask app in the background
python3 app.py &

echo "Starting secondary bash script..."
# Run another bash script in parallel
run.sh &

# Wait for both background processes to finish
wait

echo "All processes finished."
