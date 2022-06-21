# HMM-Based-Secondary-Structure-Prediction
Prediction of secondary structure of a protein, whose primary structure is given as input, by using HMM (Hidden Markov Model) and Viterbi algorithm.

# Requirements
- Python>=3.6

# Usage
- python3 hmm_based_predictor.py <training_set_path> <sequence_path> <secondary_structure_path>
- An example for sequence file, which indicates the primary structure of the protein whose secondary structure is to be predicted, is tp53_protein_sequence.txt
- An example for secondary structure file, which indicates the known secondary structure elements of the protein along with at which position intervals they occur, is tp53_secondary_structure.txt.
- If secondary structure of the protein is specified, the program shows 3x3 and individual confusion matrix computations at the end of the output.
- Specifying secondary structure of the protein is optional. If it is not specified, confusion matrix computations are not shown. So, the program can also be executed as follows:
    > python3 hmm_based_predictor.py <training_set_path> <sequence_path>
    
# Record
[![asciicast](https://asciinema.org/a/4gtBTPaeBnDd662mIXxOAR3uA.svg)](https://asciinema.org/a/4gtBTPaeBnDd662mIXxOAR3uA)