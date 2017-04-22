###
# This is just a sandbox for some char ngram features
# Function borrowed from elsewhere.
###
s creates the character n-grams like it is described in fasttext
def char_ngram_generator(text, n1=4, n2=7):
    z = []
    # Add special char for beginning, end of string
    text2 = '*'+text+'*'
    for k in range(n1,n2):
        z.append([text2[i:i+k] for i in range(len(text2)-k+1)])
    z = [ngram for ngrams in z for ngram in ngrams]
    z.append(text)
    return z
