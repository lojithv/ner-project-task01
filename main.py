# Load a spacy model and check if it has ner
import spacy
nlp=spacy.load('en_core_web_sm')

nlp.pipe_names