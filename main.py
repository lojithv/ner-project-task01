# Load a spacy model and check if it has ner
import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path

nlp = spacy.load('es_core_news_md')

# Getting the pipeline component
ner = nlp.get_pipe("ner")

text = "Una mujer de 27 años sin antecedentes, residente del hospital, " \
       "presentó odinofagia seguida de artralgia difusa y una erupción de placas eritematosas " \
       "pruriginosas extendidas, con una afectación básicamente facial y acra. El diagnóstico de " \
       "urticaria fue confirmado por un dermatólogo. No se hallaron desencadenantes excepto por las " \
       "circunstancias víricas; las pruebas séricas víricas habituales resultaron negativas. " \
       "Se estableció un tratamiento antihistamínico. 48 horas después, informó de escalofríos, " \
       "dolor torácico y fiebre de hasta 39,2 °C. Tenía linfocitopenia moderada, los análisis hepáticos " \
       "eran normales y el valor de proteína C-reactiva era de 49 mg/l. La prueba de COVID-19 " \
       "(PCR de SARS-CoV-2) dio resultado positivo. Fue tratada con paracetamol y se le mantuvieron" \
       " los antihistamínicos, con una lenta mejora de los síntomas."

doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)


################### Train Spacy NER.###########

def train_spacy():
  # Getting the pipeline component
  # ner = nlp.get_pipe("ner")

  # training data
  TRAIN_DATA = [
              ("Una mujer de 27 años sin antecedentes, residente del hospital, presentó odinofagia seguida de"
               " artralgia difusa y una erupción de placas eritematosas pruriginosas extendidas, con una afectación "
               "básicamente facial y acra. El diagnóstico de urticaria fue confirmado por un dermatólogo.",
               {"entities": [(271, 282, "PROFESION")]}),
              ]


  # Adding labels to the `ner`
  for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
      print('Adding labels to the ner...')
      ner.add_label(ent[2])

  # Disable pipeline components you dont need to change
  pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
  unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


  # TRAINING THE MODEL
  with nlp.disable_pipes(*unaffected_pipes):

    # Training for 30 iterations
    for iteration in range(30):
      print('Iteration started... ' + str(iteration))

      # shuufling examples  before every iteration
      random.shuffle(TRAIN_DATA)
      losses = {}
      # batch up the examples using spaCy's minibatch
      batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
      for batch in batches:
          texts, annotations = zip(*batch)
          nlp.update(
                      [texts],  # batch of texts
                      [annotations],  # batch of annotations
                      drop=0.5,  # dropout - make it harder to memorise data
                      losses=losses,
                  )
          print("Losses", losses)





# output doc
# MISC - Miscellaneous entities, e.g., events
# PER - ...
