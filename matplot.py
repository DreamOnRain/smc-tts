import pandas as pd

def load_data(filename):
    """Load and process data from file"""
    with open(filename, 'r') as file:
        data = [line.strip().split('|') for line in file]
    return pd.DataFrame(data, columns=['path', 'sid', 'text', 'emotion', 'language'])

# Load datasets
train_df = load_data('filelists/csemotion_train.cleaned')
test_df = load_data('filelists/csemotion_test.cleaned')

# Combine train and test for full analysis
full_df = pd.concat([train_df, test_df])

# 1. Language-Emotion Combination Statistics
def get_stats(df, name):
    """Calculate and display statistics"""
    print(f"\n=== {name} Set ===")
    
    # Language distribution
    print("\nLanguage Distribution:")
    print(df['language'].value_counts())
    
    # Emotion distribution
    print("\nEmotion Distribution:")
    print(df['emotion'].value_counts())
    
    # Combined language-emotion
    print("\nLanguage-Emotion Combination:")
    cross_tab = pd.crosstab(df['language'], df['emotion'])
    cross_tab['Total'] = cross_tab.sum(axis=1)
    cross_tab.loc['Total'] = cross_tab.sum(axis=0)
    print(cross_tab)
    
    return cross_tab

# Get statistics for each dataset
train_stats = get_stats(train_df, "Train")
test_stats = get_stats(test_df, "Test")
full_stats = get_stats(full_df, "Full")

# 2. Save results to CSV
train_stats.to_csv('train_stats.csv')
test_stats.to_csv('test_stats.csv')
full_stats.to_csv('full_stats.csv')