#!/bin/bash
cd data/qa

# Execute the Python script
python qa.py

# Now navigate to the eval directory to run metrics
cd "../../eval"

# Execute the metrics generation script
python generate.py task=qa
