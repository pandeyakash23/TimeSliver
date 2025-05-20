import numpy as np
import matplotlib.pyplot as plt

def sax_tokenizer(time_series, alphabet_size=10, word_length=1):
    # Normalize the time series
    normalized_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # Calculate the breakpoints for the alphabet
    breakpoints = np.array([-0.67, 0, 0.67])  # For alphabet_size=4
    if alphabet_size != 4:
        from scipy.stats import norm
        breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    
    # Initialize the symbolic representation
    symbolic_representation = []
    
    # Divide the time series into segments of word_length
    for i in range(0, len(normalized_series), word_length):
        segment = normalized_series[i:i + word_length]
        
        # Calculate the mean of the segment
        segment_mean = np.mean(segment)
        
        # Determine the symbol for this segment
        symbol = np.sum(segment_mean > breakpoints)
        
        # Append the symbol to the symbolic representation
        symbolic_representation.append(symbol)
    
    return symbolic_representation

# sax_representation = sax_tokenizer(time_series, alphabet_size=6, word_length=1)
