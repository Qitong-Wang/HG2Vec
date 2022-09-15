import networkx as nx
import argparse
import random
import numpy as np
import torch
import torch.multiprocessing as mp


def output_edges(args, index):
    p = 1.0 / args.p
    q = 1.0 / args.q
    # Construct Graph
    G = nx.read_edgelist(args.input_file, delimiter=",", data=[("weight", float)])
    node_list = list(G.nodes())

    print("Finish loading graph")
    # Initialize
    n_samples = len(node_list)
    n_sublist = (n_samples // args.n_process) + 1
    # https://stackoverflow.com/questions/2231663/slicing-a-list-into-a-list-of-sub-lists
    chunks =  [node_list[i:i + n_sublist] for i in range(0, n_samples, n_sublist)]
    n_samples = len(chunks[index])
    list_index = chunks[index]
    output_array = np.zeros((n_samples, args.max_length)).astype(int)
    print(output_array.shape)
    # shuffle indices
    random.shuffle(list_index)
    for n in range(0, n_samples):
        last_node = str(0)
        # if selecting all nodes, start from a random node
        if n_samples == len(node_list):
            current_node = node_list[list_index[n]]
        else:
            current_node = random.choice(node_list)
        output_array[n, 0] = current_node
        # iterate max_length times
        for i in range(1, args.max_length):
            neighbors = G.neighbors(current_node)
            neighbors_weight = list()
            for nbr in neighbors:
                if nbr == last_node:  # d = 0
                    neighbors_weight.append(p * G.edges[(current_node, nbr)]['weight'])
                    continue
                weight = G.edges[(current_node, nbr)]['weight']
                if G.has_edge(last_node, nbr):  # d = 1
                    neighbors_weight.append(weight)
                else:
                    neighbors_weight.append(q * weight)  # d = 2
            neighbors = list(G.neighbors(current_node))

            if len(neighbors) == 0:
                break
            last_node = current_node
            current_node = random.choices(neighbors, weights=neighbors_weight, k=1)[0]
            output_array[n, i] = current_node
        if n % 1000 == 0:
            print(n)
    print(output_array)
    with open(args.output_directory + str(index) + args.output_name, 'wb') as f:
        np.save(f, output_array)


def output_polar_edges(args, index):
    p = 1.0 / args.p
    q = 1.0 / args.q
    # Construct Graph
    G = nx.read_edgelist(args.input_file, delimiter=",", data=[("weight", float)])
    node_list = list(G.nodes())

    print("Finish loading graph")
    # Initialize
    n_samples = len(node_list)
    n_sublist = (n_samples // args.n_process) + 1
    # https://stackoverflow.com/questions/2231663/slicing-a-list-into-a-list-of-sub-lists
    chunks = [node_list[i:i + n_sublist] for i in range(0, n_samples, n_sublist)]
    n_samples = len(chunks[index])
    list_index = chunks[index]
    output_array = np.zeros((n_samples, args.max_length)).astype(int)
    print(output_array.shape)

    random.shuffle(list_index)
    for n in range(0, n_samples):
        last_node = str(0)
        if n_samples == len(node_list):
            current_node = node_list[list_index[n]]
        else:
            current_node = random.choice(node_list)
        output_array[n, 0] = current_node
        current_node_positive = 1
        positive_count = 1
        negative_count = 0

        for i in range(1, args.max_length):
            neighbors = G.neighbors(current_node)
            neighbors_weight = list()
            for nbr in neighbors:
                if nbr == last_node:  # d = 0
                    neighbors_weight.append(p * G.edges[(current_node, nbr)]['weight'])
                    continue
                weight = G.edges[(current_node, nbr)]['weight']
                if G.has_edge(last_node, nbr):  # d = 1
                    neighbors_weight.append(weight)
                else:
                    neighbors_weight.append(q * weight)  # d = 2

            neighbors = list(G.neighbors(current_node))

            if len(neighbors) == 0:
                break
            last_node = current_node
            indices = np.arange(len(neighbors_weight))

            current_idx = random.choices(indices, weights=np.abs(neighbors_weight), k=1)[0]
            # neighbor is an antonym of current node
            if neighbors_weight[current_idx] < 0:
                current_node_positive = current_node_positive * (-1)
            # add nodes
            if current_node_positive > 0:
                # add the node to the current line
                current_node = neighbors[current_idx]

                output_array[n, positive_count] = neighbors[current_idx]
                positive_count = positive_count + 1

            else:
                # add the node to the opposite of the current line
                current_node = neighbors[current_idx]
                output_array[n_samples + n, negative_count] = neighbors[current_idx]
                negative_count = negative_count + 1
        if n % 1000 == 0:
            print(n)
    print(output_array)
    with open(args.output_directory + str(index) + args.output_name, 'wb') as f:
        np.save(f, output_array)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Path
    argparser.add_argument('--input_file', type=str, default="./temp/edge_str_weak.csv",
                           help="The path of input edge weights")
    argparser.add_argument('--max_length', type=int, default=20,
                           help="The max length of path")
    argparser.add_argument('--n_process', type=str, default=5,
                           help="Number of processes")
    argparser.add_argument('--p', type=float, default=1.5,
                           help="The hyperparameter p")
    argparser.add_argument('--q', type=float, default=5.0,
                           help="The hyperparameter q")
    argparser.add_argument('--output_directory', type=str, default="./path/",
                           help="The path of output numerical corpus")
    argparser.add_argument('--output_name', type=str, default="_edge_str_weak.npy",
                           help="The path of output numerical corpus")
    args = argparser.parse_args()

    print("Start")
    processes = []
    for local_rank in range(5):  # + 1 for test process
        p = mp.Process(target=output_edges, args=(args, local_rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("For next task")
    args.input_file = "./temp/edge_syn_ant.csv"
    args.output_name = "_edge_syn_ant.npy"

    processes = []
    for local_rank in range(5):  # + 1 for test process
        p = mp.Process(target=output_polar_edges, args=(args, local_rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
