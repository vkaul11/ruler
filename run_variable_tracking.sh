#!/bin/bash
# Navigate to the variable tracking directory
cd data/variable_tracking

# Execute the Python script
python variable_tracking.py

# Now navigate to the eval directory to run metrics
cd ../../eval"

# Execute the metrics generation script
python generate.py task=variable_tracking
