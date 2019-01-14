# import nltk
# nltk.download()

# TODO Split into sentenses
# load data
filename = 'data/metamorphosis.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into sentences
from nltk import sent_tokenize
sentences = sent_tokenize(text)
print(sentences[0])

# TODO Split into Words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens[:100])

# TODO Filter Out Punctuation
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])

# TODO Filter out Stop Words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

# split into words
tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
words = [w for w in words if not w in stop_words]
print(words[:100])

# TODO Stem Words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in words]
print(stemmed[:100])