import os
import logging
logging.getLogger('tensorflow').disabled = True

from pytube import YouTube
from pydub import AudioSegment
import assemblyai as aai
from transformers import pipeline
import random
import string
from nltk.corpus import stopwords
import pke
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor 
import requests
import re
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet 
import nltk
from summarizer import TransformerSummarizer



def video_to_audio(yt_url):
    yt = YouTube(yt_url)
    ys = yt.streams.filter(only_audio=True).first()
    ad = ys.download()
    base, ext = os.path.splitext(ad)
    audio = AudioSegment.from_file(ad)
    audio.export(base+'.mp3', format='mp3')
    os.remove(ad)
    print("Download Complete!")
    return base+'.mp3'

def audio_to_text(filepath):
    aai.settings.api_key = "aded0399c8dc45e5b605a8856af99fd6"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filepath)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        return
    else:
        print("yahoo")
        return transcript.text

def Summary(text):
    model = TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
    result = model(text, min_length=60, max_length=500, ratio=0.4)
    summary = "".join(result)
    return summary

def extracting_keywords(text):
    print("Extracting Keywords (ProperNoun) from Fulltext...")
    keywords = []
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(text)
    pos = {'PROPN'}
    
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    
    extractor.candidate_selection(pos=pos)
    
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=15)
    for i in keyphrases:
        keywords.append(i[0])
    return keywords

def final_keywords(text, quantity):
    keywords_from_fulltext = extracting_keywords(text)
    if quantity == '0':
        print("Generating summary!!")
        generated_summary = Summary(text)
        filtered_keywords = []
        for i in keywords_from_fulltext:
            if i.lower() in generated_summary.lower():
                filtered_keywords.append(i)
        print("Selected Keywords from summary:", filtered_keywords)
        return filtered_keywords, generated_summary
    else:
        print("Selected Keywords from Full Text:", keywords_from_fulltext)
        return keywords_from_fulltext, text

def set_sentences(text):
    print("Selecting Sentences based on keywords.")
    sentences = [sent_tokenize(text)]
    sentences = [i for sent in sentences for i in sent]
    
    sentences = [sent.strip() for sent in sentences if len(sent) > 20]
    return sentences

def extract_sentences(text, quantity):
    keywords, text = final_keywords(text, quantity)
    key_processor = KeywordProcessor()
    filtered_sentences = {}
    
    for i in keywords:
        filtered_sentences[i] = []
        key_processor.add_keyword(i)
        
    sentences = set_sentences(text)

    for sent in sentences:
        keyword_searched = key_processor.extract_keywords(sent)
        for key in keyword_searched:
            filtered_sentences[key].append(sent)
    filtered_sentences = dict([(key, val) for key, val in filtered_sentences.items() if val])
    
    for i in filtered_sentences.keys():
        values = filtered_sentences[i]            
        values = sorted(values, key=len, reverse=True)
        filtered_sentences[i] = values
        
    print(filtered_sentences)
    return filtered_sentences

def wordnet_distractors(syon, word):
    distractors = []
    word = word.lower()
    original_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")      
    hypersyon = syon.hypernyms()
    if len(hypersyon) == 0:
        return distractors
    for i in hypersyon[0].hyponyms():
        name = i.lemmas()[0].name()       
        if name == original_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(i.capitalize() for i in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def conceptnet_distractors(word):
    word = word.lower()
    original_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    distractor_list = [] 
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"
    obj = requests.get(url).json()
    for edge in obj['edges']:
        link = edge['end']['term'] 
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)                 
    return distractor_list

def word_sense(sentence, keyword):
    word = keyword.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")  
    syon_sets = wordnet.synsets(word, 'n')
    if syon_sets:
        try:
            wup = max_similarity(sentence, word, 'wup', pos='n')
            adapted_lesk_output = adapted_lesk(sentence, word, pos='n')
            lowest_index = min(syon_sets.index(wup), syon_sets.index(adapted_lesk_output))
            return syon_sets[lowest_index]
        except:
            return syon_sets[0]           
    else:
        return None

def display(text, quantity):   
    filtered_sentences = extract_sentences(text, quantity)    
    options_for_mcq = {}
    for keyword in filtered_sentences:
        wordsense = word_sense(filtered_sentences[keyword][0], keyword)
        if wordsense:
           distractors = wordnet_distractors(wordsense, keyword) 
           if len(distractors) > 0:
                options_for_mcq[keyword] = distractors
           if len(distractors) < 4:
               distractors = conceptnet_distractors(keyword)
               if len(distractors) > 0:
                    options_for_mcq[keyword] = distractors                   
        else:
            distractors = conceptnet_distractors(keyword)
            if len(distractors) > 0:
                options_for_mcq[keyword] = distractors
    
    print("Creating MCQs:")
    index = 1
    for i in options_for_mcq:
        sentence = filtered_sentences[i][0]
        sentence = sentence.replace("\n", '')
        pattern = re.compile(i, re.IGNORECASE)
        output = pattern.sub(" __ ", sentence)
        print(f"{index}) {output}")
        options = [i.capitalize()] + options_for_mcq[i]
        top4 = options[:4]
        random.shuffle(top4)
        optionsno = ['a', 'b', 'c', 'd']
        for idx, choice in enumerate(top4):
            print(f"\t{optionsno[idx]}) {choice}")
        print(f"\nMore options: {options[4:8]}\n\n")
        index += 1


yt_url = input("Enter YouTube URL: ")
quantity = input("Enter 0 for summary-based keywords or 1 for full-text keywords: ")
    
audio_file = video_to_audio(yt_url)
text = audio_to_text(audio_file)
print(text)
    
if text:
    display(text, quantity)
else:
    print(".")
