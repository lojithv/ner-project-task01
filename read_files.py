import os

path = r"E:\Documents\Work\MEDDOPROF - Spain project NER\task1"

# Change the directory
os.chdir(path)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='UTF8') as f:
        print(f.read())


for file in os.listdir():

    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
        read_text_file(file_path)