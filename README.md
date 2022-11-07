# HG2Vec
This is the implemention of the paper "HG2Vec: ImprovedWord Embeddings from Dictionary and Thesaurus Based Heterogeneous Graph". Our paper is accepted by COLING 2022. Here is the link: [https://aclanthology.org/2022.coling-1.279/](https://aclanthology.org/2022.coling-1.279/) <br />
HG2Vec is a language model that learns word embeddings utilizing only dictionaries and thesauri. Our model reaches the state-of-art on multiple word similarity and relatedness benchmarks.<br />

## Running Environment:

pytorch <br />
einops <br />
numpy <br />
networkx <br />
pandas <br />
tqdm <br />
pickle <br />
csv <br />

## Directory Structures:

./input/: the folder for input dictionaries and thesauri <br />
./data/: the folder that contains id and pairs for datasets <br />
./path/: the folder contains generated paths for training <br />
./eval/: the folder contains evaluation datasets <br />

## Training:

We already include the prepossessed paths in our repo. To reproduce the results, simply run <br />
```
sh run_demo.sh 
```

However, if you want to generate the graph and paths again, you can run
```
sh data_prepossessing.sh
```

## Special Thanks:
Special thanks to Heqiong Xi, who helps the data propressessing part of this project.

