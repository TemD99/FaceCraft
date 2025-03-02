import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import os

def load_captions(filepath):
    return pd.read_csv(filepath, delimiter=':', names=['image', 'caption'])

def preprocess_captions(captions):
    # remove extra spaces and make lowercase
    captions['caption'] = captions['caption'].str.lower().str.strip()
    return captions

def get_count(captions):
    # split the captions then create a DataFrame where each word is a separate row
    words = captions['caption'].str.split().explode()
    # return the count of each word
    return Counter(words)

def plot_length(captions, output_path):
    # gets the number of words in each caption
    caption_lengths = captions['caption'].str.split().map(len)
    # plot it
    plt.figure(figsize=(8, 6))
    plt.hist(caption_lengths, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Caption Lengths', fontsize=20)
    plt.xlabel('Number of Words', fontsize=14)
    plt.ylabel('Length Frequency', fontsize=14)
    plt.savefig(os.path.join(output_path, "CaptionLengthHistogram.png"))
    plt.close()

def plot_frequency(word_counts, output_path):
    # gets the top 10 most common words
    words = []
    frequencies = []
    for word, frequency in word_counts.most_common(10):
        words.append(word)
        frequencies.append(frequency)
    # plot it
    plt.figure(figsize=(8, 6))
    plt.bar(words, frequencies, color=['lightgreen', 'lightblue','orchid'], edgecolor='black')
    plt.title('Top 10 Most Common Words', fontsize=20)
    plt.xlabel('Words', fontsize=14)
    plt.ylabel('Word Frequency', fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_path, "CaptionWordFrequencyBar.png"))
    plt.close()

def plot_wordcloud(word_counts, output_path):
    # create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    # plot it
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Caption Word Cloud', fontsize=20)
    plt.savefig(os.path.join(output_path, "CaptionWordCloud.png"))
    plt.close()

def perform_eda(filepath, output_path):
    captions = load_captions(filepath)
    captions = preprocess_captions(captions)
    word_counts = get_count(captions)
    plot_length(captions, output_path)
    plot_frequency(word_counts, output_path)
    plot_wordcloud(word_counts, output_path)

if __name__ == '__main__':
    captions = os.path.join("data/","output_captions.txt")
    output_path = os.path.join("output/")
    perform_eda(captions, output_path)
