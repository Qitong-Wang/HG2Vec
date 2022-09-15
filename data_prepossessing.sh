
# Read input files, generate Word2ID related files and the heterogeneous graph
python id_generator.py

# From teh heterogeneous graph, sample paths for training
# edge_generator.py will create 5 threads, and each thread will sample one dataset for training.
# edge_generator_dataset.py will create n thread, and each will sample 1/n dataset for training.
# python edge_generator_dataset.py
python edge_generator.py

