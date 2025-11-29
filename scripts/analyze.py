#!/usr/bin/env python3
"""
Comprehensive distant reading analysis script.
Performs various text analyses including sentiment, style, and statistical metrics.
"""

import json
import os
import re
from collections import Counter
import math


def load_text(filepath):
    """Load text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def get_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+[\s\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_words(text, lowercase=True):
    """Extract words from text."""
    # Remove punctuation and split into words
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    if lowercase:
        words = [w.lower() for w in words]
    return [w for w in words if w]


def word_frequency_analysis(words, top_n=50):
    """Analyze word frequencies."""
    # Common English stopwords
    stopwords = set([
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
        'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
        'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
        'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
        'were', 'said', 'did', 'having', 'may', 'should', 'am', 'such', 'being'
    ])

    # All words frequency
    all_freq = Counter(words)

    # Content words (excluding stopwords)
    content_words = [w for w in words if w not in stopwords]
    content_freq = Counter(content_words)

    return {
        'top_words_all': all_freq.most_common(top_n),
        'top_words_content': content_freq.most_common(top_n),
        'total_words': len(words),
        'unique_words': len(all_freq),
        'stopwords_count': len(words) - len(content_words)
    }


def get_ngrams(words, n):
    """Get n-grams from word list."""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(' '.join(words[i:i+n]))
    return ngrams


def vocabulary_richness(words):
    """Calculate vocabulary richness metrics."""
    unique_words = len(set(words))
    total_words = len(words)

    # Type-token ratio
    ttr = unique_words / total_words if total_words > 0 else 0

    # Hapax legomena (words that appear only once)
    word_freq = Counter(words)
    hapax = sum(1 for count in word_freq.values() if count == 1)
    hapax_ratio = hapax / unique_words if unique_words > 0 else 0

    # Lexical diversity (using logarithm)
    lexical_diversity = math.log(unique_words) / math.log(total_words) if total_words > 1 else 0

    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'type_token_ratio': round(ttr, 4),
        'hapax_legomena': hapax,
        'hapax_ratio': round(hapax_ratio, 4),
        'lexical_diversity': round(lexical_diversity, 4)
    }


def sentence_statistics(sentences):
    """Calculate sentence-level statistics."""
    if not sentences:
        return {'count': 0, 'avg_length': 0, 'min_length': 0, 'max_length': 0}

    lengths = [len(s.split()) for s in sentences]

    return {
        'count': len(sentences),
        'avg_length': round(sum(lengths) / len(lengths), 2),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'median_length': sorted(lengths)[len(lengths) // 2]
    }


def word_length_statistics(words):
    """Calculate word length statistics."""
    if not words:
        return {'avg_length': 0, 'min_length': 0, 'max_length': 0}

    lengths = [len(w) for w in words]

    return {
        'avg_length': round(sum(lengths) / len(lengths), 2),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'median_length': sorted(lengths)[len(lengths) // 2]
    }


def simple_pos_tagging(words):
    """Simple part-of-speech estimation based on common patterns."""
    # This is a simplified version - for production, use NLTK or spaCy
    # Common verb endings
    verb_endings = ('ed', 'ing', 'en', 'es', 's')
    # Common adjective endings
    adj_endings = ('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic')
    # Common adverb endings
    adv_endings = ('ly',)
    # Common noun endings
    noun_endings = ('tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ism', 'ship')

    pos_counts = {'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0, 'other': 0}

    for word in words:
        word_lower = word.lower()
        if word_lower.endswith(adv_endings):
            pos_counts['adverb'] += 1
        elif word_lower.endswith(adj_endings):
            pos_counts['adjective'] += 1
        elif word_lower.endswith(verb_endings):
            pos_counts['verb'] += 1
        elif word_lower.endswith(noun_endings):
            pos_counts['noun'] += 1
        else:
            pos_counts['other'] += 1

    total = sum(pos_counts.values())
    pos_distribution = {k: round(v / total * 100, 2) for k, v in pos_counts.items()}

    return {
        'counts': pos_counts,
        'distribution_percent': pos_distribution
    }


def readability_scores(text, sentences, words):
    """Calculate readability scores."""
    if not sentences or not words:
        return {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0}

    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables(word) for word in words)

    # Flesch Reading Ease
    # 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    if total_sentences > 0 and total_words > 0:
        fre = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        # Flesch-Kincaid Grade Level
        # 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
        fkg = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    else:
        fre = 0
        fkg = 0

    return {
        'flesch_reading_ease': round(fre, 2),
        'flesch_kincaid_grade': round(fkg, 2),
        'interpretation': interpret_flesch_score(fre)
    }


def count_syllables(word):
    """Estimate syllable count in a word."""
    word = word.lower()
    vowels = 'aeiou'
    syllable_count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel

    # Adjust for silent 'e'
    if word.endswith('e'):
        syllable_count -= 1

    # Every word has at least one syllable
    return max(1, syllable_count)


def interpret_flesch_score(score):
    """Interpret Flesch Reading Ease score."""
    if score >= 90:
        return "Very Easy"
    elif score >= 80:
        return "Easy"
    elif score >= 70:
        return "Fairly Easy"
    elif score >= 60:
        return "Standard"
    elif score >= 50:
        return "Fairly Difficult"
    elif score >= 30:
        return "Difficult"
    else:
        return "Very Difficult"


def simple_sentiment_analysis(text):
    """Simple sentiment analysis based on word lists."""
    # Simplified sentiment lexicon
    positive_words = set([
        'good', 'great', 'excellent', 'wonderful', 'fantastic', 'beautiful', 'love', 'happy',
        'joy', 'pleasure', 'delight', 'perfect', 'best', 'amazing', 'brilliant', 'glorious',
        'splendid', 'magnificent', 'marvelous', 'charming', 'delightful', 'cheerful', 'pleasant',
        'glad', 'grateful', 'blessed', 'fortunate', 'nice', 'kind', 'sweet', 'lovely', 'divine',
        'hope', 'hopeful', 'optimistic', 'confident', 'enthusiastic', 'excited', 'thrilled'
    ])

    negative_words = set([
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate', 'sad', 'angry',
        'pain', 'suffering', 'misery', 'unfortunate', 'dreadful', 'nasty', 'evil', 'wicked',
        'cruel', 'brutal', 'harsh', 'bitter', 'grim', 'dark', 'depressing', 'tragic', 'disaster',
        'fear', 'afraid', 'worried', 'anxious', 'nervous', 'scared', 'terrified', 'hopeless',
        'despair', 'miserable', 'wretched', 'pathetic', 'disgust', 'disgusting', 'repulsive'
    ])

    words = get_words(text, lowercase=True)

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    total_sentiment_words = positive_count + negative_count

    # Calculate sentiment score (-1 to 1)
    if total_sentiment_words > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
    else:
        sentiment_score = 0

    # Determine overall sentiment
    if sentiment_score > 0.1:
        overall_sentiment = "Positive"
    elif sentiment_score < -0.1:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {
        'score': round(sentiment_score, 4),
        'overall': overall_sentiment,
        'positive_words_count': positive_count,
        'negative_words_count': negative_count,
        'total_words_analyzed': len(words),
        'sentiment_word_ratio': round(total_sentiment_words / len(words) * 100, 2) if words else 0
    }


def extract_capitalized_entities(text):
    """Extract potential named entities (capitalized words/phrases)."""
    # Find sequences of capitalized words
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    entities = re.findall(pattern, text)

    # Filter out sentence-starting words by checking context
    # This is simplified - real NER would be more sophisticated
    entity_counter = Counter(entities)

    # Remove single-occurrence entities that are likely sentence starts
    filtered_entities = {k: v for k, v in entity_counter.items() if v > 1 or len(k.split()) > 1}

    return {
        'top_entities': sorted(filtered_entities.items(), key=lambda x: x[1], reverse=True)[:20],
        'total_unique_entities': len(filtered_entities)
    }


def analyze_text(text, label):
    """Perform comprehensive analysis on a text."""
    print(f"\nAnalyzing {label}...")

    sentences = get_sentences(text)
    words = get_words(text, lowercase=True)
    words_original_case = get_words(text, lowercase=False)

    # Perform all analyses
    word_freq = word_frequency_analysis(words)

    # Get bigrams and trigrams
    bigrams = get_ngrams(words, 2)
    trigrams = get_ngrams(words, 3)

    analysis = {
        'label': label,
        'basic_stats': {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'total_unique_words': len(set(words))
        },
        'word_frequency': word_freq,
        'common_phrases': {
            'bigrams': Counter(bigrams).most_common(20),
            'trigrams': Counter(trigrams).most_common(20)
        },
        'vocabulary_richness': vocabulary_richness(words),
        'sentence_statistics': sentence_statistics(sentences),
        'word_length_statistics': word_length_statistics(words),
        'pos_distribution': simple_pos_tagging(words_original_case),
        'readability': readability_scores(text, sentences, words),
        'sentiment': simple_sentiment_analysis(text),
        'named_entities': extract_capitalized_entities(text)
    }

    print(f"  - Words: {len(words)}")
    print(f"  - Unique words: {len(set(words))}")
    print(f"  - Sentences: {len(sentences)}")
    print(f"  - Sentiment: {analysis['sentiment']['overall']} ({analysis['sentiment']['score']})")
    print(f"  - Readability: {analysis['readability']['interpretation']}")

    return analysis


def compare_texts(analysis1, analysis2):
    """Compare two text analyses."""
    print("\nComparing texts...")

    # Get words from both texts for TF-IDF-like comparison
    words1 = set(w for w, _ in analysis1['word_frequency']['top_words_content'])
    words2 = set(w for w, _ in analysis2['word_frequency']['top_words_content'])

    distinctive_words1 = words1 - words2
    distinctive_words2 = words2 - words1
    common_words = words1 & words2

    comparison = {
        'vocabulary_comparison': {
            'text1_unique_top_words': len(distinctive_words1),
            'text2_unique_top_words': len(distinctive_words2),
            'common_top_words': len(common_words),
            'distinctive_to_text1': list(distinctive_words1)[:20],
            'distinctive_to_text2': list(distinctive_words2)[:20]
        },
        'statistics_comparison': {
            'vocabulary_richness': {
                'text1_ttr': analysis1['vocabulary_richness']['type_token_ratio'],
                'text2_ttr': analysis2['vocabulary_richness']['type_token_ratio'],
                'difference': round(analysis1['vocabulary_richness']['type_token_ratio'] -
                                    analysis2['vocabulary_richness']['type_token_ratio'], 4)
            },
            'sentence_length': {
                'text1_avg': analysis1['sentence_statistics']['avg_length'],
                'text2_avg': analysis2['sentence_statistics']['avg_length'],
                'difference': round(analysis1['sentence_statistics']['avg_length'] -
                                    analysis2['sentence_statistics']['avg_length'], 2)
            },
            'word_length': {
                'text1_avg': analysis1['word_length_statistics']['avg_length'],
                'text2_avg': analysis2['word_length_statistics']['avg_length'],
                'difference': round(analysis1['word_length_statistics']['avg_length'] -
                                    analysis2['word_length_statistics']['avg_length'], 2)
            },
            'readability': {
                'text1_flesch': analysis1['readability']['flesch_reading_ease'],
                'text2_flesch': analysis2['readability']['flesch_reading_ease'],
                'difference': round(analysis1['readability']['flesch_reading_ease'] -
                                    analysis2['readability']['flesch_reading_ease'], 2)
            },
            'sentiment': {
                'text1_score': analysis1['sentiment']['score'],
                'text2_score': analysis2['sentiment']['score'],
                'text1_overall': analysis1['sentiment']['overall'],
                'text2_overall': analysis2['sentiment']['overall']
            }
        },
        'pos_overlay_data': {
            'text1': analysis1['pos_distribution']['distribution_percent'],
            'text2': analysis2['pos_distribution']['distribution_percent']
        }
    }

    print(f"  - Common top words: {len(common_words)}")
    print(f"  - Distinctive words (text1): {len(distinctive_words1)}")
    print(f"  - Distinctive words (text2): {len(distinctive_words2)}")

    return comparison


def main():
    """Main analysis pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load preprocessed texts
    dostoyevsky_path = os.path.join(base_dir, 'data/processed/dostoyevsky_notes_from_underground.txt')
    chernyshevsky_path = os.path.join(base_dir, 'data/processed/chernyshevsky_what_is_to_be_done.txt')

    print("Loading texts...")
    dostoyevsky_text = load_text(dostoyevsky_path)
    chernyshevsky_text = load_text(chernyshevsky_path)

    # Analyze both texts
    dostoyevsky_analysis = analyze_text(dostoyevsky_text, "Dostoyevsky - Notes from the Underground")
    chernyshevsky_analysis = analyze_text(chernyshevsky_text, "Chernyshevsky - What Is To Be Done?")

    # Compare texts
    comparison = compare_texts(dostoyevsky_analysis, chernyshevsky_analysis)

    # Combine all results
    results = {
        'dostoyevsky': dostoyevsky_analysis,
        'chernyshevsky': chernyshevsky_analysis,
        'comparison': comparison,
        'metadata': {
            'analysis_version': '1.0',
            'texts_analyzed': 2,
            'analysis_types': [
                'word_frequency', 'vocabulary_richness', 'sentence_statistics',
                'word_length', 'pos_distribution', 'readability', 'sentiment',
                'named_entities', 'comparative_analysis'
            ]
        }
    }

    # Save to JSON
    output_path = os.path.join(base_dir, 'results/analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Analysis complete! Results saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
