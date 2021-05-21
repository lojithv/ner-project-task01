# Load a spacy model and check if it has ner
import spacy

nlp = spacy.load('es_core_news_md')

# nlp.pipe_names

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


# output doc
# MISC - Miscellaneous entities, e.g., events
# PER - ...
