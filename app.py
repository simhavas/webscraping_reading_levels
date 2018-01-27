# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:10:00 2018

@author: simha
"""


import re, requests, math
from flask import Flask, request, jsonify
#from collections import Counter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
#import MySQLdb

app = Flask(__name__)

specialSyllables_en = """tottered 2
chummed 1
peeped 1
moustaches 2
shamefully 3
messieurs 2
satiated 4
sailmaker 4
sheered 1
disinterred 3
propitiatory 6
bepatched 2
particularized 5
caressed 2
trespassed 2
sepulchre 3
flapped 1
hemispheres 3
pencilled 2
motioned 2
poleman 2
slandered 2
sombre 2
etc 4
sidespring 2
mimes 1
effaces 2
mr 2
mrs 2
ms 1
dr 2
st 1
sr 2
jr 2
truckle 2
foamed 1
fringed 2
clattered 2
capered 2
mangroves 2
suavely 2
reclined 2
brutes 1
effaced 2
quivered 2
h'm 1
veriest 3
sententiously 4
deafened 2
manoeuvred 3
unstained 2
gaped 1
stammered 2
shivered 2
discoloured 3
gravesend 2
60 2
lb 1
unexpressed 3
greyish 2
unostentatious 5
"""

fallback_cache = {}

fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou",
                   "sia$", ".ely$"]

fallback_addsyl = ["ia", "riet", "dien", "iu", "io", "ii",
                   "[aeiouy]bl$", "mbl$",
                   "[aeiou]{3}",
                   "^mc", "ism$",
                   "(.)(?!\\1)([aeiouy])\\2l$",
                   "[^l]llien",
                   "^coad.", "^coag.", "^coal.", "^coax.",
                   "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
                   "dnt$"]


# Compile our regular expressions
for i in range(len(fallback_subsyl)):
    fallback_subsyl[i] = re.compile(fallback_subsyl[i])
for i in range(len(fallback_addsyl)):
    fallback_addsyl[i] = re.compile(fallback_addsyl[i])

def _normalize_word(word):
    return word.strip().lower()

# Read our syllable override file and stash that info in the cache
for line in specialSyllables_en.splitlines():
    line = line.strip()
    if line:
        toks = line.split()
        assert len(toks) == 2
        fallback_cache[_normalize_word(toks[0])] = int(toks[1])

# function to count number of words
def count(word):
    word = _normalize_word(word)
    if not word:
        return 0

    # Check for a cached syllable count
    count = fallback_cache.get(word, -1)
    if count > 0:
        return count

    # Remove final silent 'e'
    if word[-1] == "e":
        word = word[:-1]

    # Count vowel groups
    count = 0
    prev_was_vowel = 0
    for c in word:
        is_vowel = c in ("a", "e", "i", "o", "u", "y")
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Add & subtract syllables
    for r in fallback_addsyl:
        if r.search(word):
            count += 1
    for r in fallback_subsyl:
        if r.search(word):
            count -= 1

    # Cache the syllable count
    fallback_cache[word] = count

    return count





SPECIAL_CHARS = ['.', ',', '!', '?']

# function to count number of characters
def get_char_count(words):
    characters = 0
    for word in words:
        #characters += len(word.decode("utf-8"))
        characters += len(word)
    return characters


# function to extract words from the text
def get_words(text=''):
    words = []
    words = word_tokenize(text)
    return words

# function to extrac sentences from the text
def get_sentences(text=''):
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenize(text)
    return sentences

# function to count syllables
def count_syllables(words):
    syllableCount = 0
    for word in words:
        syllableCount += count(word)
    return syllableCount

# a model to extract complex words but it is not perfect
def count_complex_words(text=''):
    words = get_words(text)
    sentences = get_sentences(text)
    complex_words = 0
    found = False
    cur_word = []
    
    for word in words:          
        cur_word.append(word)
        if count_syllables(cur_word)>= 3:
            
            #Checking proper nouns. If a word starts with a capital letter
            #and is NOT at the beginning of a sentence we don't add it
            #as a complex word.
            if not(word[0].isupper()):
                complex_words += 1
            else:
                for sentence in sentences:
                    if (sentence).startswith(word):
                        found = True
                        break
                if found: 
                    complex_words += 1
                    found = False
                
        cur_word.remove(word)
    return complex_words



# function which extracts all the required elements for calculating the metrics
def analyze_text(text):
    words = get_words(text)
    char_count = get_char_count(words)
    word_count = len(words)
    sentence_count = len(get_sentences(text))
    syllable_count = count_syllables(words)
    complexwords_count = count_complex_words(text)
    avg_words_p_sentence = word_count/sentence_count
    
    analyzedVars = {
        'words': words,
        'char_cnt': float(char_count),
        'word_cnt': float(word_count),
        'sentence_cnt': float(sentence_count),
        'syllable_cnt': float(syllable_count),
        'complex_word_cnt': float(complexwords_count),
        'avg_words_p_sentence': float(avg_words_p_sentence)
    }
    
    return analyzedVars

# Automated Readability Index
def ARI(analyzedVars):
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        score = 4.71 * (analyzedVars['char_cnt'] / analyzedVars['word_cnt']) + 0.5 * (analyzedVars['word_cnt'] / analyzedVars['sentence_cnt']) - 21.43
    return score

# Flesch Reading Ease Index
def FleschReadingEase(analyzedVars):
        score = 0.0 
        if analyzedVars['word_cnt'] > 0.0:
            score = 206.835 - (1.015 * (analyzedVars['avg_words_p_sentence'])) - (84.6 * (analyzedVars['syllable_cnt']/ analyzedVars['word_cnt']))
        return round(score, 4)

# Flesch-Kincaid Grade Level Index
def FleschKincaidGradeLevel(analyzedVars):
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        score = 0.39 * (analyzedVars['avg_words_p_sentence']) + 11.8 * (analyzedVars['syllable_cnt']/ analyzedVars['word_cnt']) - 15.59
    return round(score, 4)

# Gunning Fog Index
def GunningFogIndex(analyzedVars):
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        score = 0.4 * ((analyzedVars['avg_words_p_sentence']) + (100 * (analyzedVars['complex_word_cnt']/analyzedVars['word_cnt'])))
    return round(score, 4)
# Smog Index
def SMOGIndex(analyzedVars):
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        score = (math.sqrt(analyzedVars['complex_word_cnt']*(30/analyzedVars['sentence_cnt'])) + 3)
    return score

# Coleman-Liau Index
def ColemanLiauIndex(analyzedVars):
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        score = (5.89*(analyzedVars['char_cnt']/analyzedVars['word_cnt']))-(30*(analyzedVars['sentence_cnt']/analyzedVars['word_cnt']))-15.8
    return round(score, 4)

# Lix Index
def LIX(analyzedVars):
    longwords = 0.0
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        for word in analyzedVars['words']:
            if len(word) >= 7:
                longwords += 1.0
        score = analyzedVars['word_cnt'] / analyzedVars['sentence_cnt'] + float(100 * longwords) / analyzedVars['word_cnt']
    return score
# RIX Index
def RIX(analyzedVars):
    longwords = 0.0
    score = 0.0 
    if analyzedVars['word_cnt'] > 0.0:
        for word in analyzedVars['words']:
            if len(word) >= 7:
                longwords += 1.0
        score = longwords / analyzedVars['sentence_cnt']
    return score

@app.route('/link/<path:url>',methods = ['GET'])
def analyze(url):
#    errors = []
    results = []
    
    #url = request.form['url']
    #urllink = urllib.request.Request('http://www.python.org/fish.html')
    r = requests.get(url)
    if r:
        try:
#            labels = []
#            values = []
            proc_text = []
            #news_open = urllib.urlopen(r).read()
            news_soup = BeautifulSoup(r.text, "html.parser")
            #news_soup = unicodeddata.normalize("NKFD",news_soup)
            news_para = news_soup.find_all("p", text = True)
            for item in news_para:
                # SPLIT WORDS, JOIN WORDS TO REMOVE EXTRA SPACES
                para_text = (' ').join((item.find_all(text=True)))
                para_text = (' ').join((item.text).split())
        
                # COMBINE LINES/PARAGRAPHS INTO A LIST
                #proc_text.append(para_text.encode('utf-8'))
                proc_text.append(para_text)
                
            #proc_text.remove('')
            proc_text = " ".join(proc_text)
            
            x = analyze_text(proc_text)
            ARI_score = ARI(x)
            FRE_score = FleschReadingEase(x)
            FKG_score = FleschKincaidGradeLevel(x)
            GFI_score = GunningFogIndex(x)
            SMIdx = SMOGIndex(x)
            CLIdx = ColemanLiauIndex(x)
            LIX_score = LIX(x)
            RIX_score = RIX(x)
#            results = ARI_score
#            labels = ["ARI_Score","FRE_Score","FKG_Score","GFI_Score","SmogIdx","ColemanIdx","LIX","RIX"]
#            values = [ARI_score,FRE_score,FKG_score,GFI_score,SMIdx,CLIdx,LIX_score,RIX_score]
            
#            
            results.append({"ARI_Score": ARI_score,"FRE_Score": FRE_score,"FKG_Score": FKG_score,"GFI_Score":GFI_score,"SmogIdx": SMIdx,"ColemanIdx":CLIdx,"LIX":LIX_score,"RIX": RIX_score })
            return jsonify(results)
            
        except:
            return jsonify({'message': "Unable to get URL. Please make sure it's valid and try again."})
        
    else:
        return jsonify({'message': "Unable to get URL. Please make sure it's valid and try again."})
            






if __name__ == '__main__':
    app.run(debug=True)
    
    