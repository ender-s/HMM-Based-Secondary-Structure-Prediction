import os, sys, io, math

class SequenceReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def set_file_path(self, file_path):
        self.file_path = file_path
    
    def get_file_path(self):
        return self.file_path

    def read_sequence(self):
        with open(self.file_path) as f:
            lines = f.read().strip().splitlines()
        sequence = None

        for line in lines:
            if not (line.startswith(">") or line.startswith("#")):
                sequence = line
                break
            elif line.startswith(">"):
                sequence_descriptor = line
        return sequence_descriptor, sequence

class Utils:
    @staticmethod
    def find_max_index(l):
        max_index = 0
        i = 1
        while i < len(l):
            if l[i] > l[max_index]:
                max_index = i
            i = i + 1
        return max_index
    
    @staticmethod
    def content_to_dict(content):
        l = [i.strip() for i in content.splitlines() if i.strip()]
        return {key: value for key, value in [((int(i.split()[1])-1, int(i.split()[2]) - 1), i[0][0].replace("S", "E")) for i in l]}
    
    @staticmethod
    def count_for_confusion_matrix(truth_dict, prediction_dict, truth_key, prediction_key):
        start = min(truth_dict.keys())
        end = max(truth_dict.keys())
        counter = 0

        for i in range(start, end + 1):
            if prediction_dict[i] == prediction_key and truth_dict[i] == truth_key:
                counter += 1
        return counter
    
    @staticmethod
    def count_individual_confusion_statistics(truth_dict, prediction_dict, key):
        start = min(truth_dict.keys())
        end = max(truth_dict.keys())

        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

        for i in range(start, end + 1):
            if truth_dict[i] == key and prediction_dict[i] == key: true_positive += 1
            if truth_dict[i] != key and prediction_dict[i] != key: true_negative += 1
            if truth_dict[i] != key and prediction_dict[i] == key: false_positive += 1
            if truth_dict[i] == key and prediction_dict[i] != key: false_negative += 1

        return true_positive, true_negative, false_positive, false_negative
    
    @staticmethod
    def path_to_position_dict(path):
        return {key: value for key, value in [(index, path[index]) for index in range(len(path))]}
    
    @staticmethod
    def generate_position_dict(d, length):
        result = {}
        sorted_keys = sorted(d)
        i = 0
        for interval in sorted_keys:
            ll, ul = interval
            if i < ll:
                for y in range(i, ll):
                    result[y] = 'N'
            for y in range(ll, ul + 1):
                result[y] = d[interval]
            i = ul + 1
        if i < length:
            for y in range(i, length):
                result[y] = 'N'
        return result

class ViterbiAlgorithm:
    def __init__(self, hmm, sequence):
        self.hmm = hmm
        self.sequence = sequence
        self.column_count = len(self.sequence)
        self.states_list = self.hmm.get_states()
        self.matrix = [[0 for j in range(len(sequence))] for i in range(len(self.states_list))]
        
        self.arrow_map = {}
        self.fill_in_the_matrix()

    def fill_in_the_matrix(self):
        j = 0
        for i in range(len(self.states_list)):
            state = self.states_list[i]
            self.matrix[i][j] = self.hmm.tlp('START', state) + self.hmm.elp(state, self.sequence[j])
        
        for j in range(1, self.column_count):
            aa = self.sequence[j] # aa stands for amino_acid
            for i in range(len(self.states_list)):
                state = self.states_list[i]
                self.matrix[i][j] = self.hmm.elp(state, aa)
                list_to_look_for_max = []
                for k in range(len(self.states_list)):
                    inner_state = self.states_list[k]
                    list_to_look_for_max.append(self.matrix[k][j - 1] + self.hmm.tlp(inner_state, state))
                max_index = Utils.find_max_index(list_to_look_for_max)
                self.arrow_map[(i, j)] = max_index
                self.matrix[i][j] += list_to_look_for_max[max_index]

                if j == self.column_count - 1: # if we are in the last column, take into account the end state probability
                    self.matrix[i][j] += self.hmm.tlp(state, 'END')

    def construct_path(self):
        self.path = ""
        list_to_look_for_max = []
        for i in range(len(self.states_list)):
            list_to_look_for_max.append(self.matrix[i][self.column_count - 1])
        max_index = Utils.find_max_index(list_to_look_for_max)

        j = self.column_count - 1
        i = max_index
        log_probability = list_to_look_for_max[max_index]
        while j > 0:
            to_go = self.arrow_map[(i, j)]
            self.path = self.states_list[i] + self.path
            i = to_go
            j -= 1
        self.path = self.states_list[i] + self.path
        return self.path, log_probability

class HMM:
    def __init__(self, training_set_path):
        self.load_training_set(training_set_path)
        self.preprocess_training_set()

        # X and the lowercase letters are for the letters found in the training set
        self.amino_acid_alphabet = "ACDEFGHIKLMNPQRSTVWYXabcdegfhijklmnopqrutvw"
        
        self.states = {'H': {key: 0 for key in self.amino_acid_alphabet}, 
                       'E': {key: 0 for key in self.amino_acid_alphabet}, 
                       'T': {key: 0 for key in self.amino_acid_alphabet}}

        self.transitions = {}

        for state_i in "HET":
            for state_j in "HET":
                self.transitions[(state_i, state_j)] = 0
        
        for state in "HET":
            self.transitions[("START", state)] = 0
        
        for state in "HET":
            self.transitions[(state, "END")] = 0

        self.train()

    def get_states(self):
        return tuple("HET")

    def tlp(self, from_state, to_state):
        # tlp stands for transition_log_probability
        return self.transitions[(from_state, to_state)]
    
    def elp(self, state, amino_acid):
        # elp stands for emission_log_probability
        return self.states[state][amino_acid]

    def load_training_set(self, training_set_path):
        with open(training_set_path) as file:
            training_set = file.read().strip().splitlines()
        self.training_sequences = {}
        
        index_list = [i for i in range(len(training_set)) if training_set[i].startswith(">")]
        for index in index_list:
            self.training_sequences[training_set[index].strip()] = (training_set[index + 1].strip(), training_set[index + 2].strip())
        
        print(f"Loaded {len(self.training_sequences)} training samples.")        

    def preprocess_training_set(self):
        print("Preprocessing training data...", end = ' ')
        sys.stdout.flush()
        for key, sequence_structure_tuple in self.training_sequences.items():
            sequence, structure = sequence_structure_tuple
            preprocessed_sequence_io = io.StringIO()
            preprocessed_structure_io = io.StringIO()
            for i in range(len(sequence)):
                structure_char = structure[i]
                sequence_char = sequence[i]
                if structure_char != "_":
                    preprocessed_sequence_io.write(sequence_char)
                    if structure_char in ('G', 'H', 'I'):
                        preprocessed_structure_io.write('H')
                    elif structure_char in ('B', 'E'):
                        preprocessed_structure_io.write('E')
                    elif structure_char in ('T', 'S', 'L'):
                        preprocessed_structure_io.write('T')

            self.training_sequences[key] = (preprocessed_sequence_io.getvalue(), preprocessed_structure_io.getvalue())
        print("Done!")

    def train(self):
        print ("Training...", end = ' ')
        sys.stdout.flush()
        inner_transition_counts = {'H': 0, 'E': 0, 'T': 0}
        start_transition_count = 0

        for key, sequence_structure_tuple in self.training_sequences.items():
            sequence, structure = sequence_structure_tuple
            
            for index in range(len(sequence)):
                sequence_char = sequence[index]
                structure_char = structure[index]
                if index == 0:
                    start_transition_count += 1
                    self.transitions[('START', structure_char)] += 1
                else:
                    inner_transition_counts[structure[index - 1]] += 1
                    self.transitions[(structure[index - 1], structure_char)] += 1
                    if index == len(sequence) - 1:
                        inner_transition_counts[structure_char] += 1
                        self.transitions[(structure_char, 'END')] += 1

                self.states[structure_char][sequence_char] += 1

        for state, emissions in self.states.items():
            summation = sum(emissions.values())
            for amino_acid, count in emissions.items():
                self.states[state][amino_acid] = math.log2((count + 1) / (summation + len(self.amino_acid_alphabet)))
        
        for state_i in "HET":
            for state_j in "HET":
                self.transitions[(state_i, state_j)] = math.log2(self.transitions[(state_i, state_j)] / inner_transition_counts[state_i])

        for state in "HET":
            self.transitions[("START", state)] = math.log2(self.transitions[("START", state)] / start_transition_count)
        
        for state in "HET":
            self.transitions[(state, "END")] = math.log2(self.transitions[(state, "END")] / inner_transition_counts[state])

        print("Done!")

class Main:
    def __init__(self):
        try:
            training_set_path = sys.argv[1]
            sequence_path = sys.argv[2]
        except IndexError:
            self.print_usage()
            sys.exit()

        truth_interval_dict = None
        if len(sys.argv) > 3:
            secondary_structure_path = sys.argv[3]
            with open(secondary_structure_path) as f:
                truth_interval_dict = Utils.content_to_dict(f.read().strip())

        sequence_reader = SequenceReader(sequence_path)
        header, sequence = sequence_reader.read_sequence()
        self.hmm = HMM(training_set_path)
        self.viterbi_algorithm = ViterbiAlgorithm(self.hmm, sequence)
        path, log_probability = self.viterbi_algorithm.construct_path()
        print("\nInput protein sequence:\n" + "-"*30 + "\n" + header + "\n" + sequence)
        print("\nThe path predicted by HMM:\n" + "-"*30 + "\n" + path)
        print("\nLog2 probability of this path:\n" + "-"*30 + "\n" + str(log_probability))

        if truth_interval_dict:
            truth_dict = Utils.generate_position_dict(truth_interval_dict, len(sequence))
            prediction_dict = Utils.path_to_position_dict(path)

            print("\n3x3 confusion matrix computations:")
            print("True".ljust(10), "Predicted".ljust(10), "Count".ljust(10))
            for key_i in "HET":
                for key_j in "HET":
                    print (key_i.ljust(10), key_j.ljust(10), str(Utils.count_for_confusion_matrix(truth_dict, prediction_dict, key_i, key_j)).ljust(10))
            
            print("Individual confusion matrix computations:")
            for key in "HET":
                print(f"Individual confusion matrix computations for {key}:")
                print("TP".ljust(10), "TN".ljust(10), "FP".ljust(10), "FN".ljust(10))
                tp, tn, fp, fn = Utils.count_individual_confusion_statistics(truth_dict, prediction_dict, key)
                print(str(tp).ljust(10), str(tn).ljust(10), str(fp).ljust(10), str(fn).ljust(10))

    def print_usage(self):
        print(f"Usage: python3 {os.path.split(sys.argv[0])[-1]} <training_set_path> <sequence_path> <secondary_structure_path>")

if __name__ == "__main__":
    main = Main()
