#!/usr/bin/env python
# coding: utf-8

# In[6]:


import openfst_python as fst
import math
import numpy as np
import pandas as pd

# In[10]:


def parse_lexicon(lex_file,original):
    """
    Parse the lexicon file and return it in dictionary form.
    
    Args:
        lex_file (str): filename of lexicon file with structure '<word> <phone1> <phone2>...'
                        eg. peppers p eh p er z

    Returns:
        lex (dict): dictionary mapping words to list of phones
    """
    
    lex = {}  # create a dictionary for the lexicon entries (this could be a problem with larger lexica)
    # there can be duplicate phones for the same word 
    with open(lex_file, 'r') as f:
        for line in f:
            line = line.split()  # split at each space
            word = line[0]
            phones = line[1:]
            if original == False:
                if (word in lex):
                    lex[word].append(phones)
                else:
                    lex[word] = [phones]
            else:
                lex[line[0]] = [line[1:]]  # first field the word, the rest is the phones
#     print(f"lex: {lex}")
    return lex


# In[3]:


def generate_symbols_table(lexicon, n=3):
    '''
    Return word, phone and state symbol tables based on the supplied lexicon
        
    Args:
        lexicon (dict): lexicon to use, created from the parse_lexicon() function
        n (int): number of states for each phone HMM
        
    Returns:
        word_table (fst.SymbolTable): table of words
        phone_table (fst.SymbolTable): table of phones
        state_table (fst.SymbolTable): table of HMM phone-state IDs
    '''
    
    state_table = fst.SymbolTable()
    phone_table = fst.SymbolTable()
    word_table = fst.SymbolTable()
    
    # add empty <eps> symbol to all tables
    state_table.add_symbol('<eps>')
    phone_table.add_symbol('<eps>')
    word_table.add_symbol('<eps>')
    
    for word, phone_list  in lexicon.items():
        
        word_table.add_symbol(word)
        
        for phones in phone_list: # for each phone
            for p in phones:
#                 print(f"Adding phone {p} from word {word} to phone table")
                phone_table.add_symbol(p)
                for i in range(1,n+1): # for each state 1 to n
                    state_table.add_symbol('{}_{}'.format(p, i))
    # <!> DEBUGS
#     for word in word_table:
#         print(word)
#     print('word_table: {}\nphone_table: {}\nstate_table: {}'.format(list(word_table), list(phone_table), list(state_table)))
    return word_table, phone_table, state_table


def generate_output_table(word_table, phone_table):
    output_table = fst.SymbolTable()
    for phone_id, phone_str in list(phone_table):
        output_table.add_symbol(phone_str)
    for word_id, word_str in list(word_table):
        output_table.add_symbol(word_str)
    return output_table
    
# In[4]:


# def parse_lexicon(lex_file):
#     """
#     Parse the lexicon file and return it in dictionary form.
    
#     Args:
#         lex_file (str): filename of lexicon file with structure '<word> <phone1> <phone2>...'
#                         eg. peppers p eh p er z

#     Returns:
#         lex (dict): dictionary mapping words to list of phones
#     """
    
#     lex = {}  # create a dictionary for the lexicon entries (this could be a problem with larger lexica)
#     with open(lex_file, 'r') as f:
#         for line in f:
#             line = line.split()  # split at each space
#             lex[line[0]] = [line[1:]]  # first field the word, the rest is the phones
#     return lex


# In[5]:


# def generate_symbol_tables(lexicon, n=3):
#     '''
#     Return word, phone and state symbol tables based on the supplied lexicon
        
#     Args:
#         lexicon (dict): lexicon to use, created from the parse_lexicon() function
#         n (int): number of states for each phone HMM
        
#     Returns:
#         word_table (fst.SymbolTable): table of words
#         phone_table (fst.SymbolTable): table of phones
#         state_table (fst.SymbolTable): table of HMM phone-state IDs
#     '''
    
#     state_table = fst.SymbolTable()
#     phone_table = fst.SymbolTable()
#     word_table = fst.SymbolTable()
    
#     # add empty <eps> symbol to all tables
#     state_table.add_symbol('<eps>')
#     phone_table.add_symbol('<eps>')
#     word_table.add_symbol('<eps>')
    
#     for word, phones  in lexicon.items():
        
#         word_table.add_symbol(word)
        
#         for p in phones: # for each phone
            
#             phone_table.add_symbol(p)
#             for i in range(1,n+1): # for each state 1 to n
#                 state_table.add_symbol('{}_{}'.format(p, i))
            
#     return word_table, phone_table, state_table


# In[7]:


def generate_phone_wfst(f, start_state, phone, n, state_table, phone_table, weight_fwd, weight_self):
    """
    Generate a WFST representing an n-state left-to-right phone HMM.
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        phone (str): the phone label 
        n (int): number of states of the HMM
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
        
    Returns:
        the final state of the FST
    """
    
    current_state = start_state
    
    for i in range(1, n+1):
        
        in_label = state_table.find('{}_{}'.format(phone, i))
        
        sl_weight = None if weight_self==None else fst.Weight('log', -math.log(weight_self))  # weight for self-loop
        next_weight = None if weight_fwd==None else fst.Weight('log', -math.log(weight_fwd)) # weight for forward
            
        # self-loop back to current state
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        
        # transition to next state
        
        # we want to output the phone label on the final state
        # note: if outputting words instead this code should be modified
        if i == n:
            out_label = phone_table.find(phone)
        else:
            out_label = 0   # output empty <eps> label
            
        next_state = f.add_state()
#         next_weight = fst.Weight('log', -math.log(0.9)) # weight to next state
        f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))    
       
        current_state = next_state
    return current_state




# In[9]:


def generate_word_wfst(f, start_state, word, n):
    """ Generate a WFST for any word in the lexicon, composed of n-state phone WFSTs.
        This will currently output phone labels.
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        word (str): the word to generate
        n (int): states per phone HMM
        
    Returns:
        the constructed WFST
    
    """

    current_state = start_state
    
    # iterate over all the phones in the word
    for phone in lex[word]:   # will raise an exception if word is not in the lexicon
        # your code here
        
        current_state = generate_phone_wfst(f, current_state, phone, n, state_table, phone_table)
    
        # note: new current_state is now set to the final state of the previous phone WFST
        
    f.set_final(current_state)
    
    return f


# In[11]:


def generate_phone_recognition_wfst(n, state_table, phone_table):
    """ generate a HMM to recognise any single phone in the lexicon
    
    Args:
        n (int): states per phone HMM

    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst()
    
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    # get a list of all the phones in the lexicon
    # there are lots of way to do this.  Here, we use the set() object

    # will contain all unique phones in the lexicon
    phone_set = set()
    
    for pronunciation in lex.values():
        phone_set = phone_set.union(pronunciation)
        
    for phone in phone_set:
        
        # we need to add an empty arc from the start state to where the actual phone HMM
        # will begin.  If you can't see why this is needed, try without it!
        current_state = f.add_state()
        f.add_arc(start_state, fst.Arc(0, 0, None, current_state))
    
        end_state = generate_phone_wfst(f, current_state, phone, n, state_table, phone_table)
    
        f.set_final(end_state)

    return f


# In[12]:


def generate_phone_sequence_recognition_wfst(n, state_table, phone_table):
    """ generate a HMM to recognise any single phone sequence in the lexicon
    
    Args:
        n (int): states per phone HMM

    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst()
    
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    phone_set = set()
    
    for pronunciation in lex.values():
        phone_set = phone_set.union(pronunciation)
        
    for phone in phone_set:
        current_state = f.add_state()
        f.add_arc(start_state, fst.Arc(0, 0, None, current_state))
    
        end_state = generate_phone_wfst(f, current_state, phone, n, state_table, phone_table)
        
        f.add_arc(end_state, fst.Arc(0,0, None, start_state))
        f.set_final(end_state)

    return f


# In[14]:


def generate_word_sequence_recognition_wfst(n, lex, original=False, weight_fwd=None, weight_self=None):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
    Returns:
        the constructed WFST
    
    """
    if (weight_fwd!=None and weight_self!=None):
        f = fst.Fst('log')
        none_weight = fst.Weight('log', -math.log(1))
    else:
        f = fst.Fst()
        none_weight = None
    lex = parse_lexicon(lex, original)
    word_table, phone_table, state_table = generate_symbols_table(lex,3)
    output_table = generate_output_table(word_table, phone_table)
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    # make fst
    for word, phone_list in lex.items():            
        for phones in phone_list:
            initial_state = f.add_state()
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), none_weight, initial_state))
            current_state = initial_state
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, weight_fwd, weight_self)                
            
            f.set_final(current_state)
            f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table

def generate_word_sequence_recognition_wfst_bigram(n, lex, original=False, weight_fwd=None, weight_self=None):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
    Returns:
        the constructed WFST
    
    """
    if (weight_fwd!=None and weight_self!=None):
        f = fst.Fst('log')
        none_weight = fst.Weight('log', -math.log(1))
    else:
        f = fst.Fst()
        none_weight = None
    lex = parse_lexicon(lex, original)
    word_table, phone_table, state_table = generate_symbols_table(lex,3)
    output_table = generate_output_table(word_table, phone_table)
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    # -- dictionaries for initial and last states
    dict_initial = {}
    dict_final = {}
    # make fst
    for word, phone_list in lex.items():            
        for phones in phone_list:
            initial_state = f.add_state()
            # -- add to initial dict
            if word in dict_initial:
                dict_initial[word].append(initial_state)
            else:
                dict_initial[word] = [initial_state]
            # -- add arcs
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), none_weight, initial_state))
            current_state = initial_state
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, weight_fwd, weight_self)                
            f.set_final(current_state)
            f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
            # -- add to final dict
            if word in dict_final:
                dict_final[word].append(current_state)
            else:
                dict_final[word] = [current_state]
    # -- add bidirectional arcs 
    for word, last_state_list in dict_final.items():                  # list of final states 4 word
        for last_state in last_state_list:                            # final state from lsit
            for word_bi, initial_state_list in dict_initial.items():  # list of initial satates
                for initial_state in initial_state_list:              # state from list
                    f.add_arc(last_state, fst.Arc(0, output_table.find(word_bi), none_weight, initial_state))
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table



def generate_word_sequence_recognition_wfst_test(n, lex, original=False, weight_fwd=None, weight_self=None):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
    Returns:
        the constructed WFST
    
    """
    if (weight_fwd!=None and weight_self!=None):
        f = fst.Fst('log')
        none_weight = fst.Weight('log', -math.log(1))
    else:
        f = fst.Fst()
        none_weight = None
        
    lex = parse_lexicon(lex, original)
    
    word_table, phone_table, state_table = generate_symbols_table(lex,3)
    output_table = generate_output_table(word_table, phone_table)
#     print('output_table: {}'.format(list(output_table)))
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    # -- make fst
    for word, phone_list in lex.items():
        for phones in phone_list:
            initial_state = f.add_state()
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), none_weight, initial_state))
            current_state = initial_state
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, weight_fwd, weight_self)
            f.set_final(current_state)
#             f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table


def get_word_occurences(transcript_files):
    """ Gets the occurences of the words in the transcripts
    
        Args:
            transcript_files (list): list of paths to transcript txt files
            
        Returns:
            dict_first (dict): word -> number of times word was said first
            dict_last (dict): word -> number of times word was said last
            dict_all (dict): word -> number of times word was said
    """
    # count how many words are at the start
    dict_first = {}   # dictionary of first words
    dict_last  = {}   # dictionary of last words
    dict_all   = {}   # dictionary of all words
    txt_all    = []   # list of all sentences
    for txt in transcript_files[:]:
        txt_file = open(txt, "r")       # open file
        txt_str = txt_file.read()       # string from file
        txt_list = txt_str.split()      # separate @ white space
        txt_first = txt_list[0]     # first word
        txt_last  = txt_list[-1]    # last word
        # -- add to dictionary & lists
        txt_all.append(txt_str)           # add to list of strings
        if txt_first in dict_first:       # add first words
            dict_first[txt_first] += 1
        else:
            dict_first[txt_first] = 1

        if txt_last in dict_last:         # add last words
            dict_last[txt_last] += 1
        else:
            dict_last[txt_last] = 1

        for word in txt_list:             # add all words
            if word in dict_all:
                dict_all[word] += 1
            else:
                dict_all[word] = 1
    return (dict_first, dict_last, dict_all)

def get_bigram_df(transcript_files):
    """ Gets the bigram dataframe of the words in the transcripts
    
        Args:
            transcript_files (list): list of paths to transcript txt files
            
        Returns:
            bigram_df (pd.DataFrame): dataframe with columns next word and rows previous word
    """
    list_labels = ['a', 'of', 'peck', 'peppers', 'peter', 'picked', 'pickled', 'piper', 'the', "where's"]
    txt_all = []
    for txt in transcript_files[:]:
        txt_file = open(txt, "r")
        txt_str = txt_file.read() 
        txt_all.append(txt_str)
    
    bigrams = [b for l in txt_all[:] for b in zip(l.split()[:-1], l.split()[1:])]
    bigram_dict = {}
    for bigram in bigrams:
        if (bigram in bigram_dict):
            bigram_dict[bigram] += 1
        else:
            bigram_dict[bigram] = 1
            
    columns = list_labels
    data = np.zeros([len(columns),len(columns)])
    for (col, row),occurence in bigram_dict.items():
        col_idx = columns.index(col)
        row_idx = columns.index(row)
        data[col_idx,row_idx] = occurence
    
    bigram_df = pd.DataFrame(data, columns,columns)
    bigram_df = pd.concat(
        [pd.concat(
            [bigram_df],
            keys=['Word After'], axis=1)],
        keys=['Word Before'])
    
    return bigram_df 

def calculate_probabilities(words_occurrence_dict):
    """ Calculates the probabilities given the occurce of the words
        
    Args:
        words_occurrence_dict (dict): word -> occurance of word in transcripts
        
    Returns:
        probability_dict (dict): word -> probability of word
    """
    total = sum(words_occurrence_dict.values())
    probability_dict = {}
    for word, occurrence in words_occurrence_dict.items():
        prob = occurrence/total
#         print(f'occurrence: {occurrence}, total: {total}, prob: {prob}')
        probability_dict[word] = prob
    return probability_dict


def get_bigram_prob_df(bigram_df, dict_all):
    """Gets the bigram probability dataframe
    
        Args:
            bigram_df (pd.DataFrame): dataframe with columns as words after and rows as words before
            dict_all (dict): Dictionary containing the counts of all the words in the transcripts
            
        Returns:
            pd.DataFrame: Bigram probabilities in a dataframe with probability(w|w-1) being df['Word After',w]['Word Before',w-1]
    """
    # make a bigram probability table
    bigram_prob_df = bigram_df.copy(deep=True)
    for idx_col, column in enumerate(bigram_prob_df.columns[:]):
        for idx_row, row in enumerate(bigram_prob_df[column]):
            counts_of_before = dict_all[bigram_prob_df.columns[idx_row][1]]
            counts_before_after = bigram_prob_df.iloc[idx_row][idx_col]
            try:
                prob = counts_before_after/counts_of_before
            except e:
                print(f"Excpetion {e}")
            bigram_prob_df.iloc[idx_row][idx_col] = prob
    return bigram_prob_df


def generate_WFST_final_probability(n, lex, weight_fwd, weight_self, weights_final, original=False):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
        weight_final (dict): word -> probability of final state
    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst('log')
    none_weight = fst.Weight('log', -math.log(1))

    lex = parse_lexicon(lex, original)
    
    word_table, phone_table, state_table = generate_symbols_table(lex,3)
    output_table = generate_output_table(word_table, phone_table)

    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    for word, phone_list in lex.items():
        for phones in phone_list:
            initial_state = f.add_state()
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), none_weight, initial_state))
            current_state = initial_state
            
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, weight_fwd, weight_self)
        
            f.set_final(current_state)
            f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
        
        # final word state should be current state
        prob = weights_final[word]
        weight = fst.Weight('log', -math.log(prob))
        f.set_final(current_state, weight)
#         print(f"Current state: {current_state} for word {word} is prob {prob} with log prob{(weight)}")
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table 

def generate_WFST_unigram(n, lex, weight_fwd, weight_self, weights_start, original=False):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
        weights_start (dict): word -> probability of word
    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst('log')
    none_weight = fst.Weight('log', -math.log(1))

    lex = parse_lexicon(lex, original)
    
    word_table, phone_table, state_table = generate_symbols_table(lex,3)
    output_table = generate_output_table(word_table, phone_table)

    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    for word, phone_list in lex.items():
        for phones in phone_list:
            initial_state = f.add_state()
            
            # initial state is the start of the word and hence will have probability going to it
            prob = weights_start[word]
            weight = fst.Weight('log', -math.log(prob))
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), weight, initial_state))
#             print(f"Current state: {initial_state} for word {word} is prob {prob} with log prob{(weight)}")
            # -------
            current_state = initial_state
            
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, weight_fwd, weight_self)
        
            f.set_final(current_state)
            f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table 

## -- Silent States -- ##

def generate_silent_phone_wfst(f, start_state, state_table, phone_table):
    """
    Generate a WFST representing an n-state left-to-right phone HMM.
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        phone (str): the phone label 
        n (int): number of states of the HMM
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
        
    Returns:
        the final state of the FST
    """
#     print(f"states before silent: {list(f.states())}")
    current_state = start_state
                                
    # start with creating the states
    n = 5
    for i in range(1, n+1):
        current_state = f.add_state()                   
    
    WFST_silent = list(f.states())[-(n+1):]
    
    # manually make the ergodic topology
    s0 = WFST_silent[0]
    s0_label = state_table.find('sil_1')
    
    s1 = WFST_silent[1]
    s1_label = state_table.find('sil_2')
    
    s2 = WFST_silent[2]
    s2_label = state_table.find('sil_3')
    
    s3 = WFST_silent[3]
    s3_label = state_table.find('sil_4')
    
    s4 = WFST_silent[4]
    s4_label = state_table.find('sil_5')
   
    
    # create arcs
    # s0
    f.add_arc(s0, fst.Arc(s0_label, 0, fst.Weight('log',-math.log(0.5)), s0))
    f.add_arc(s0, fst.Arc(s0_label, 0, fst.Weight('log',-math.log(0.5)), s1))
    # s1
    f.add_arc(s1, fst.Arc(s1_label, 0, fst.Weight('log',-math.log(1/3.0)), s1))
    f.add_arc(s1, fst.Arc(s1_label, 0, fst.Weight('log',-math.log(1/3.0)), s2))
    f.add_arc(s1, fst.Arc(s1_label, 0, fst.Weight('log',-math.log(1/3.0)), s3))
    #s2
    f.add_arc(s2, fst.Arc(s2_label, 0, fst.Weight('log',-math.log(1/3.0)), s1))
    f.add_arc(s2, fst.Arc(s2_label, 0, fst.Weight('log',-math.log(1/3.0)), s2))
    f.add_arc(s2, fst.Arc(s2_label, 0, fst.Weight('log',-math.log(1/3.0)), s3))
    #s3
    f.add_arc(s3, fst.Arc(s3_label, 0, fst.Weight('log',-math.log(1/4.0)), s1))
    f.add_arc(s3, fst.Arc(s3_label, 0, fst.Weight('log',-math.log(1/4.0)), s2))
    f.add_arc(s3, fst.Arc(s3_label, 0, fst.Weight('log',-math.log(1/4.0)), s3))
    f.add_arc(s3, fst.Arc(s3_label, 0, fst.Weight('log',-math.log(1/4.0)), s4))
    # s4
    f.add_arc(s4, fst.Arc(s4_label, 0, fst.Weight('log',-math.log(0.5)), s4))
    f.add_arc(s4, fst.Arc(s4_label, 0, fst.Weight('log',-math.log(0.5)), current_state))
    
#     print(f"silent states: {WFST_silent}")
    return current_state


def generate_WFST_silent(n, lex, weight_fwd, weight_self, original=False):
    """ generate a HMM to recognise any single word sequence for words in the lexicon and includes a silence state
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon
        weight_fwd (int): weight value
        weight_self (int): weight value of self node
        weights_start (dict): word -> probability of word
    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst('log')
    none_weight = fst.Weight('log', -math.log(1))

    original_lex = parse_lexicon(lex, original)
    # add the silent states
    silent_word = '<silence>'
    silent_phones = ['sil_0','sil_1','sil_2','sil_3','sil_4','sil_5']
    silence_lex = original_lex.copy()
    silence_lex[silent_word] = [silent_phones] # makes sure output table contains it
    # -----
#     print(f"lex: {silence_lex}")
    word_table, phone_table, state_table = generate_symbols_table(original_lex,3)
    word_table.add_symbol(silent_word)
    for phone in silent_phones:
        state_table.add_symbol(phone)
    phone_table.add_symbol('sil')
    
#     print(f'state table: {list(state_table)}')
    output_table = generate_output_table(word_table, phone_table)

    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    # skip silent phones by using original lex 
    for word, phone_list in original_lex.items():
        for phones in phone_list:
            initial_state = f.add_state()
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), none_weight, initial_state))
            current_state = initial_state
            
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, weight_fwd, weight_self)
        
            f.set_final(current_state)
            f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
    
    # need to add the silent state seperately
    current_state = f.add_state()
    f.add_arc(start_state, fst.Arc(0, output_table.find(silent_word), none_weight, current_state))
    current_state = generate_silent_phone_wfst(f, current_state, state_table, output_table)
    f.set_final(current_state)
    f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
    
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table 



# class TrieNode: 
      
#     # Trie node class 
#     def __init__(self, phone, state): 
#         self.children = []
#         self.phone = phone
#         self.state = state
        
#     def add_child(self, trie_node):
#         self.children.append(trie_node)
        
#     def __eq__(self, val):
#         return (val == self.phone)
    
#     def __repr__(self):
#         return self.phone
        
    
# class Trie:
    
#     def __init__(self, lex, f):
#         self.root = self.getNode('') 
#         self.lex = lex
        
#         lex = parse_lexicon(lex, original)
#         self.word_table, self.phone_table, self.state_table = generate_symbols_table(self.lex,1)
#         self.output_table = generate_output_table(self.word_table, self.phone_table)
        
#         start_state = self.f.add_state()
#         self.f.set_start(start_state)
        
#         for word in self.lex:
#             self.insert(word)
  
#     def getNode(self, phone): 
#         # Returns new trie node (initialized to NULLs) 
#         return TrieNode(phone) 
    
    
#     def insert(self,word):
#         current_node = self.root
#         current_phone = self.lex[word][0]
#         self._insert(word, current_node, current_phone)
    
#     def _insert(self, word, node, phone):
#         # check if the current node is this phone
#         found = False
#         phones = self.lex[word]
        
#         for child_node in node.children: 
#             if (child_node.phone == phone):
#                 found = True
#                 if (phone == phones[-1]):
#                     self.f.set_final(node)
#                     return node
#                 else:
#                     next_phone = phones[phones.index(phone) + 1]
#                     next_node = self._insert(word, child_node, next_phone)
#                     f.add_arc(node.state, fst.Arc(0, output_table.find(phone), None, next_node))
                    
                    
                
#         # create the new phone node
#         if (found == False):
#             new_state = self.f.add_state()
#             new_node = self.getNode(phone, new_state)
#             node.add_child(new_node)
            
#             # if this is last phone then return
#             if (phone == phones[-1]):
#                 self.f.set_final(node)
#                 return node
#             else:
#                 next_phone = phones[phones.index(phone) + 1]
#                 next_node = self._insert(word, new_node, next_phone)
#                 f.add_arc(node.state, fst.Arc(0, output_table.find(phone), None, next_node))

#     def __repr__(self):
#         node = self.root
#         string = f"{node}"
#         string = self._toString(string, node)
#         return string
        
#     def _toString(self, string, node):
#         string = f"{string} + {node.phone}"
#         for child_node in node.children:
#             string += self._toString(string, child_node)
#         return string
                
        

