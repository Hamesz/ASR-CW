#!/usr/bin/env python
# coding: utf-8

# In[6]:


import openfst_python as fst
import math

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


def generate_phone_wfst(f, start_state, phone, n, state_table, phone_table, log):
    """
    Generate a WFST representing an n-state left-to-right phone HMM.
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        phone (str): the phone label 
        n (int): number of states of the HMM
        
    Returns:
        the final state of the FST
    """
    
    current_state = start_state
    
    for i in range(1, n+1):
        
        in_label = state_table.find('{}_{}'.format(phone, i))
        
        if (log):
            sl_weight = fst.Weight('log', -math.log(0.1))  # weight for self-loop
            next_weight = fst.Weight('log', -math.log(0.9))
        else:
            sl_weight = None
            next_weight = None
            
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


def generate_word_sequence_recognition_wfst(n, lex, original=False, log=True):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM
        original (bool): True/False - origianl/optimized lexicon

    Returns:
        the constructed WFST
    
    """
    if (log):
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
    
    
    
    for word, phone_list in lex.items():
        for phones in phone_list:
            initial_state = f.add_state()
            f.add_arc(start_state, fst.Arc(0, output_table.find(word), none_weight, initial_state))
            current_state = initial_state
            
            for phone in phones:
                current_state = generate_phone_wfst(f, current_state, phone, n, state_table, output_table, log)
        
            f.set_final(current_state)
            f.add_arc(current_state, fst.Arc(0, 0, none_weight, start_state))
        
    f.set_input_symbols(state_table)
    f.set_output_symbols(output_table)
    return f, word_table


# In[ ]:




