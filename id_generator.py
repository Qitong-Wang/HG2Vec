import numpy as np
import pandas as pd
import argparse
import pickle
import os
import math
import csv
import statistics

global_word_frequency = dict()
word2id = dict()
word2id["[TEMP]"] = 0
id_info = dict()
id_info[0] = [0, "[TEMP]"]  # Frequency, word
strong_pair = dict()
weak_pair = dict()
syn_pair = dict()
ant_pair = dict()


def get_id(query_word):
    if query_word in word2id.keys():
        return word2id[query_word]
    else:
        return -1


# Read definition pairs. Update id_info immediately after reading. The min count is 1.
def read_info(file_path):
    input_file = open(file_path, "r")
    lines = input_file.readlines()
    for line in lines:
        if (len(line) <= 3):  # Special case of input "A.D." and empty
            continue
        words = line.split()
        if (len(words)!=2):
            continue
        global_word_frequency[words[0]] = global_word_frequency.get(words[0], 0) + 1
        global_word_frequency[words[1]] = global_word_frequency.get(words[1], 0) + 1


def read_pairs(file_path, target_dictionary, undirect=True):
    input_file = open(file_path, "r")
    longest_length = 0
    lines = input_file.readlines()
    for line in lines:
        if (len(line) <= 3):  # Special case of input "A.D." and empty
            continue
        words = line.split()
        if (len(words)!=2):
            continue
        word1_index = get_id(words[0])
        word2_index = get_id(words[1])
        # Does not contain words.
        if word1_index == -1 or word2_index == -1:
            continue
        # Add word1 word2 pair
        if word1_index in target_dictionary.keys():
            # Does not contain words.
            if word2_index in target_dictionary[word1_index]:
                continue
            target_dictionary[word1_index].append(word2_index)
            if len(target_dictionary[word1_index]) > longest_length:
                longest_length = len(target_dictionary[word1_index])
        else:
            target_dictionary[word1_index] = list()
            target_dictionary[word1_index].append(word2_index)
            if len(target_dictionary[word1_index]) > longest_length:
                longest_length = len(target_dictionary[word1_index])
        if undirect:  # Undirected pair. Add word2 word1 pair
            if word2_index in target_dictionary.keys():
                if word1_index in target_dictionary[word2_index]:
                    continue
                target_dictionary[word2_index].append(word1_index)
                if len(target_dictionary[word2_index]) > longest_length:
                    longest_length = len(target_dictionary[word2_index])
            else:
                target_dictionary[word2_index] = list()
                target_dictionary[word2_index].append(word1_index)
                if len(target_dictionary[word2_index]) > longest_length:
                    longest_length = len(target_dictionary[word2_index])

    print("Pairs dictionary contains key " + str(len(target_dictionary)))
    print("Pairs dictionary longest value " + str(longest_length))


def update_id_info(min_count, dictionary):
    for w, c in dictionary.items():  # w is word, c is count
        if c < min_count:
            continue
        if w in word2id.keys():
            idx = word2id[w]
            id_info[idx][0] = id_info[idx][0] + c
        else:
            wid = len(word2id)  # 0 is [TEMP]
            word2id[w] = wid
            id_info[wid] = [c, w]

    print("Total unique word embeddings: " + str(len(word2id)))


def output_file(input_file, output_file):
    # Generate outputs
    num_output = []
    for line in open(input_file, encoding="utf8"):
        line = line.split()
        for word in line:
            if len(word) > 0:
                if word in word2id.keys():
                    num_output.append(word2id[word])
    print("Output file length: " + str(len(num_output)))

    with open(output_file[:-4] + ".npy", 'wb') as f:
        np.save(f, np.array(num_output))


def create_tfidf_dictionary(input_file):
    tf_dictionary = dict()
    N_dictionary = dict()
    # Convert to idx
    document = pd.read_csv(input_file, header=None)
    total_N = document.shape[0]
    for n in range(0, total_N):
        key_word = document.iloc[n, 0]
        key_idx = word2id[key_word]
        line = document.iloc[n, 1].split()
        line_idx = [word2id[w] for w in line]
        # Store tf data
        tf_freq = 1.0 / len(line_idx)
        for word_idx in line_idx:
            tf_dictionary[(key_idx, word_idx)] = tf_dictionary.get((key_idx, word_idx), 0.0) + tf_freq

        set_index = set(line_idx)
        for word_idx in set_index:
            N_dictionary[word_idx] = N_dictionary.get(word_idx, 0) + 1
    # Calculate tfidf
    tfidf_dictionary = dict()
    for key, value in tf_dictionary.items():
        w1 = key[0]
        w2 = key[1]
        tfidf = value * math.log(total_N / N_dictionary[w2])
        tfidf_dictionary[(w1, w2)] = tfidf_dictionary.get((w1, w2), 0.0) + tfidf
        tfidf_dictionary[(w2, w1)] = tfidf_dictionary.get((w2, w1), 0.0) + tfidf
    return tfidf_dictionary



def add_pair_into_dictionary(pair_dictionary, idx_dictionary, weight):
    for key, value in pair_dictionary.items():
        for v in value:
            if key == v:
                continue
            idx_dictionary[(key, v)] = idx_dictionary.get((key, v), 0.0) + weight
            a = 1
            # tfidf_dictionary[(v, key)] = tfidf_dictionary.get((v, key), 0.0) + weight
    return idx_dictionary



def output_csv(output_file):
    strong_weight = 0.8
    weak_weight = 0.4
    print("Strong weight: {}".format(strong_weight))
    print("Weak weight: {}".format(weak_weight))
    str_weak_dictionary = dict()
    str_weak_dictionary = add_pair_into_dictionary(strong_pair, str_weak_dictionary, strong_weight)
    str_weak_dictionary = add_pair_into_dictionary(weak_pair, str_weak_dictionary, weak_weight)
    synonym_weight = 1.0
    ant_weight = -1.0
    synonym_dictionary = dict()
    synonym_dictionary = add_pair_into_dictionary(syn_pair, synonym_dictionary, synonym_weight)
    synonym_dictionary = add_pair_into_dictionary(ant_pair, synonym_dictionary, ant_weight)

    f = open(output_file[:-4] + "_str_weak" + output_file[-4:], 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    for key, value in str_weak_dictionary.items():
        output_list = [key[0], key[1], value]
        writer.writerow(output_list)
    f.close()
    print("Output strong weak edges number: " + str(len(str_weak_dictionary)))
    f = open(output_file[:-4] + "_syn_ant" + output_file[-4:], 'w', encoding='UTF8', newline='')
    writer = csv.writer(f)
    for key, value in synonym_dictionary.items():
        output_list = [key[0], key[1], value]
        writer.writerow(output_list)
    f.close()
    print("Output syn ant edges number: " + str(len(synonym_dictionary)))


def output_pairs(args):
    # https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file
    with open(args.output_id_info, 'w') as f:
        for key in id_info.keys():
            f.write("%s,%s,%s\n" % (key, id_info[key][0], id_info[key][1]))

    output_file = open(args.output_strong_file, "wb")
    pickle.dump(strong_pair, output_file)
    output_file.close()

    output_file = open(args.output_weak_file, "wb")
    pickle.dump(weak_pair, output_file)
    output_file.close()

    output_file = open(args.output_syn_file, "wb")
    pickle.dump(syn_pair, output_file)
    output_file.close()

    output_file = open(args.output_ant_file, "wb")
    pickle.dump(ant_pair, output_file)
    output_file.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Path

    argparser.add_argument('--input_folder', type=str, default="./input/files/",
                           # default="./input/files/", #default="None",
                           help="The path of input corpus or definitions folder")
    argparser.add_argument('--input_strong_file', type=str, default="./input/strong-pairs.txt",
                           help="The path of input strong pairs")
    argparser.add_argument('--input_weak_file', type=str, default="./input/weak-pairs.txt",
                           help="The path of input weak pairs")
    argparser.add_argument('--input_syn_file', type=str, default="./input/syn-pairs.txt",
                           help="The path of input strong pairs")
    argparser.add_argument('--input_ant_file', type=str, default="./input/ant-pairs.txt",
                           help="The path of input weak pairs")
    argparser.add_argument('--output_file', type=str, default="./temp/edge.csv",
                           help="The path of output numerical corpus or edges in csv")
    argparser.add_argument('--output_folder', type=str, default="./temp/",
                           help="The path of output folder")
    argparser.add_argument('--output_id_info', type=str, default="./data/id_info.csv",
                           help="The path of output id info")
    argparser.add_argument('--output_strong_file', type=str, default="./data/strong-pairs.pkl",
                           help="The path of input strong pairs")
    argparser.add_argument('--output_weak_file', type=str, default="./data/weak-pairs.pkl",
                           help="The path of input weak pairs")
    argparser.add_argument('--output_syn_file', type=str, default="./data/syn-pairs.pkl",
                           help="The path of input strong pairs")
    argparser.add_argument('--output_ant_file', type=str, default="./data/ant-pairs.pkl",
                           help="The path of input weak pairs")
    argparser.add_argument('--min_count', type=int, default=1,
                           help="Min count of input")
    argparser.add_argument('--t1', action='store_true')
    argparser.add_argument('--t2', action='store_true')
    args = argparser.parse_args()

    # create temp folder if not exists
    temp_exist = os.path.exists("./temp")
    if not temp_exist:
        os.makedirs("./temp")

    print("---Read file---")
    # args.input_folder is None:
    read_info(args.input_strong_file)
    read_info(args.input_weak_file)
    read_info(args.input_syn_file)
    read_info(args.input_ant_file)

    update_id_info(args.min_count, global_word_frequency)
    print("---Read strong pairs---")
    read_pairs(args.input_strong_file, strong_pair, undirect=True)
    print("---Read weak pairs---")
    read_pairs(args.input_weak_file, weak_pair, undirect=True)
    print("---Read synonym pairs---")
    read_pairs(args.input_syn_file, syn_pair, undirect=True)
    print("---Read antonym pairs---")
    read_pairs(args.input_ant_file, ant_pair, undirect=True)
    print("---Output files---")

    output_csv(args.output_file)
    output_pairs(args)
