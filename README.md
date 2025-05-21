## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:Named Entity Recognition (NER) tools often lack user-friendly interfaces, hindering accessibility for rapid text analysis. Integrating complex NLP models into practical, interactive applications presents a significant challenge. There is a clear need for an intuitive prototype that demonstrates and leverages advanced NER capabilities effectively. This project addresses the gap by providing a readily usable solution.

### DESIGN STEPS:

#### STEP 1:
Install Libraries: Ensure transformers and pipeline are installed.
#### STEP 2:
Import Libraries: Import pipeline from transformers.
#### STEP 3:
Load Model: Load a pre-trained NER model (e.g., "dslim/bert-large-NER") using the pipeline function.
#### STEP 4:
Authenticate (if needed): If you're running this in a Google Colab or similar environment, authenticate with your Hugging Face token.
#### STEP 5:
Configure Model: Set up configuration details like batch size, storage, and expected input/output.
#### STEP 6:
Load Input Text:
Option A: Sample Text: Click the "Load Sample Text from file" button.
Option B: Upload File: Upload a .txt or .pdf file.
Option C: Manual Input: Type your text directly into the "Input Text" area.
#### STEP 7:
Process Text: The system will then process the input text using the loaded NER model.
#### STEP 8:
Display Entities: The recognized entities will be displayed in the "Output Entities" section.
#### STEP 9:
Launch Interface: Use demo.launch() to start the Gradio web interface for the prototype.
### PROGRAM:
```
!pip install gradio transformers torch pdfplumber --quiet
import gradio as gr
from transformers import pipeline
import pdfplumber

ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def load_sample_text():
    import os
    if not os.path.exists("sample_text.txt"):
        return "sample_text.txt not found. Please upload it first."
    with open("sample_text.txt", "r", encoding="utf-8") as f:
        return f.read()

def read_file(file):
    if file is None:
        return ""
    filename = file.name.lower()
    if filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file.name) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        return "Unsupported file format. Please upload a .txt or .pdf file."

def ner_text(text):
    if not text.strip():
        return "Please enter text."
    entities = ner_pipeline(text)
    result = []
    for ent in entities:
        result.append(f"{ent['entity_group']}: '{ent['word']}' (score: {ent['score']:.2f})")
    return "\n".join(result)

def update_textbox_with_file(file):
    if file is None:
        return ""
    return read_file(file)

with gr.Blocks() as demo:
    gr.Markdown("# Named Entity Recognition (NER) Prototype")
    
    textbox = gr.Textbox(label="Input Text", lines=8, placeholder="Type text or use buttons below")
    load_button = gr.Button("Load Sample Text from file")
    ner_button = gr.Button("Run NER")
    output = gr.Textbox(label="Named Entities", lines=10)
    
    load_button.click(fn=load_sample_text, outputs=textbox)
    file_input.change(fn=update_textbox_with_file, inputs=file_input, outputs=textbox)
    ner_button.click(fn=ner_text, inputs=textbox, outputs=output)

demo.launch()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/8e1d636a-b228-4c8d-a2ed-dd725d40a560)
![image](https://github.com/user-attachments/assets/05fbb091-ac75-40de-ae70-f3c599e59b88)

### RESULT:
Therefore the program is excuted successfully
