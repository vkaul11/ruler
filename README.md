Steps for running the Ruler Eval
1. Install packages in requirements.txt
2. Change the default.yaml files in each of the tasks as needed. In particular, change max_seq_length to 64k, 32k etc according to the context length you want.
   a. https://github.com/vkaul11/ruler/blob/main/data/niah/conf/simulation/default.yaml
   b. https://github.com/vkaul11/ruler/blob/main/data/qa/conf/simulation/default.yaml
   c. https://github.com/vkaul11/ruler/blob/main/data/variable_tracking/conf/simulation/default.yaml
   d. https://github.com/vkaul11/ruler/blob/main/data/hash_hop/conf/config.yaml
3. Change the model_id, auth_key and url for evaluation https://github.com/vkaul11/ruler/blob/main/eval/conf/config.yaml
4. Run the bash scripts https://github.com/vkaul11/ruler/blob/main/run_all_tasks.sh for running all the 3 tasks that will print out the metric per example and average metric or Run 1)NIAH task https://github.com/vkaul11/ruler/blob/main/run_niah.sh
   or 2) Variable tracking task https://github.com/vkaul11/ruler/blob/main/run_variable_tracking.sh
   or 3) QA task https://github.com/vkaul11/ruler/blob/main/run_qa.sh if you want to run the tasks individually and get the metrics
   or a different bash file
   4) https://github.com/vkaul11/ruler/blob/main/run_hashhop.sh
5. The eval directory will also have the predictions and errors for each of the tasks outputted.
