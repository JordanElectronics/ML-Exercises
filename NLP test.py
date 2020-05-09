from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

sentences2 = [
    'I love my dog',
    'I love my cat',
    'I love my bird'
    'This is a sentence that is long'
]

#"bird" is unknown to the tokenizer. The "<OOV>" is used in its place.
sequences2 = tokenizer.texts_to_sequences(sentences2)
print(sequences2)

padded = pad_sequences(sequences2)
print('Padded:')
print(padded)
padded = pad_sequences(sequences2, padding='post', truncating='post', maxlen=5)
print(padded)
