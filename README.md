![image](https://github.com/mytechnotalent/pa/blob/main/Personal%20Assistant.png?raw=true)

<br>

# Personal Assistant
A Personal Assistant leveraging Retrieval-Augmented Generation (RAG) and the LLaMA-3.1-8B-Instant Large Language Model (LLM). This tool is designed to revolutionize PDF document analysis tasks by combining machine learning with retrieval-based systems.

## Origin of the RAG Architecture

Retrieval-Augmented Generation (RAG) is a powerful technique in natural language processing (NLP) that combines retrieval-based methods with generative models to produce more accurate and contextually relevant outputs. This approach was introduced in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Facebook AI Research (FAIR).

For further reading and a deeper understanding of RAG, refer to the original paper by Facebook AI Research: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401). 

## Overview of the RAG Architecture

The RAG model consists of two main components:
1. **Retriever**: This component retrieves relevant documents from a large corpus based on the input query.
2. **Generator**: This component generates responses conditioned on the retrieved documents.

## Mathematical Formulation

### Retriever

The retriever selects the top $k$ documents from a corpus $\mathcal{D}$ based on their relevance to the input query $q$.

### Generator

The generator produces a response $r$ based on the input query $q$ and the retrieved documents $\{d_1, d_2, \ldots, d_k\}$.

### Combining Retriever and Generator

The final probability of generating a response $r$ given the query $q$ is obtained by marginalizing over the top $k$ retrieved documents:

$$
P(r \mid q) = \sum_{i=1}^{k} P(d_i \mid q) P(r \mid q, d_i)
$$

Here, $P(d_i \mid q)$ is the normalized relevance score of document $d_i$ given the query $q$, and $P(r \mid q, d_i)$ is the probability of generating response $r$ given the query $q$ and document $d_i$.

## Implementation Details

### Training

The RAG model is trained in two stages:
1. **Retriever Training**: The retriever is trained to maximize the relevance score $s(q, d_i)$ for relevant documents.
2. **Generator Training**: The generator is trained to maximize the probability $P(r \mid q, d_i)$ for the ground-truth responses.

### Inference

During inference, the RAG model retrieves the top $k$ documents for a given query and generates a response conditioned on these documents. The final response is obtained by marginalizing over the retrieved documents as described above.

## Conclusion

RAG leverages the strengths of both retrieval-based and generation-based models to produce more accurate and informative responses. By conditioning the generation on retrieved documents, RAG can incorporate external knowledge from large corpora, leading to better performance on various tasks.

The combination of retriever and generator in the RAG model makes it a powerful approach for tasks that require access to external knowledge and the ability to generate coherent and contextually appropriate responses.

### Install Conda Environment
1. To select a Conda environment in Visual Studio Code, press the play button in the next cell which will open up a command prompt then select `Python Environments...`
2. A new command prompt will pop up and select `+ Create Python Environment`.
3. A new command prompt will again pop up and select `Conda Creates a .conda Conda environment in the current workspace`.
4. A new command prompt will again pop up and select `* Python 3.11`.


```python
!conda create -n pa python=3.11 -y
```

### !!! ACTION ITEM !!!
In order for the Conda environment to be available, you need to close down VSCode and reload it and select `rea` in the Kernel area in the top-right of VSCode.
1. In the VSCode pop-up command window select `Select Another Kernel...`.
2. In the next command window select `Python Environments...`.
3. In the next command window select `pa (Python 3.11.9)`.

### Install Packages


```python
!conda install -n pa \
    pytorch \
    torchvision \
    torchaudio \
    cpuonly \
    -c pytorch \
    -c conda-forge \
    --yes
%pip install -U ipywidgets
%pip install -U requests
%pip install -U llama-index
%pip install -U llama-index-embeddings-huggingface
%pip install -U llama-index-llms-groq
%pip install -U groq
%pip install -U gradio
```

### Install Tesseract


```python
import os
import platform
import subprocess
import requests


def install_tesseract():
    """
    Installs Tesseract OCR based on the operating system.
    """
    os_name = platform.system()
    if os_name == "Linux":
        print("Detected Linux. Installing Tesseract using apt-get...")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr"], check=True)
    elif os_name == "Darwin":
        print("Detected macOS. Installing Tesseract using Homebrew...")
        subprocess.run(["brew", "install", "tesseract"], check=True)
    elif os_name == "Windows":
        tesseract_installer_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
        installer_path = "tesseract-ocr-w64-setup-5.4.0.20240606.exe"
        response = requests.get(tesseract_installer_url)
        with open(installer_path, "wb") as file:
            file.write(response.content)
        tesseract_path = r"C:\Program Files\Tesseract-OCR"
        os.environ["PATH"] += os.pathsep + tesseract_path
        try:
            result = subprocess.run(["tesseract", "--version"], check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running Tesseract: {e}")
    else:
        print(f"Unsupported OS: {os_name}")


install_tesseract()
```

### Convert PDF to OCR


```python
import webbrowser

url = "https://www.ilovepdf.com/ocr-pdf"
webbrowser.open_new(url)
```

### Import Libraries


```python
import os
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import gradio as gr
```

### !!! ACTION ITEM !!!
Visit https://console.groq.com/keys and set up an API Key then replace `<GROQ_API_KEY>` below with the newly generated key.


```python
os.environ["GROQ_API_KEY"] = "<GROQ_API_KEY>"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

### Disable Tokenizer Parallelism Globally


```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Data Ingestion


```python

def load_single_pdf_from_directory(directory_path):
    """
    Load a single PDF file from the specified directory.

    Args:
        directory_path (str): The path to the directory containing the PDF files.

    Returns:
        documents (list): The loaded documents if exactly one PDF is found.
    """
    pdf_files = [file for file in os.listdir(directory_path) if file.lower().endswith(".pdf")]
    if len(pdf_files) != 1:
        print("Error: There must be exactly one PDF file in the directory.")
        return None
    file_path = os.path.join(directory_path, pdf_files[0])
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    return documents


directory_path = "files"
documents = load_single_pdf_from_directory(directory_path)
if documents is not None:
    print(f"Successfully loaded {len(documents)} pages(s).")
else:
    print("ABORTING NOTEBOOK!")

```

    Successfully loaded 2 pages(s).


### Chunking


```python
text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(documents)
```

### Embedding Model


```python
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Define LLM Model


```python
llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
```

### Configure Service Context


```python
Settings.embed_model = embed_model
Settings.llm = llm
```

### Create Vector Store Index


```python
# Debug VectorStoreIndex
print("VectorStoreIndex initialization")
vector_index = VectorStoreIndex.from_documents(
    documents, 
    show_progress=True, 
    node_parser=nodes
)
```

    VectorStoreIndex initialization



    Parsing nodes:   0%|          | 0/2 [00:00<?, ?it/s]



    Generating embeddings:   0%|          | 0/10 [00:00<?, ?it/s]


#### Persist/Save Index


```python
vector_index.storage_context.persist(persist_dir="./storage_mini")
```

#### Define Storage Context


```python
storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
```

#### Load Index


```python
index = load_index_from_storage(storage_context)
```

### Define Query Engine


```python
query_engine = index.as_query_engine()
```

#### Feed in user query


```python
def query_function(query):
    """
    Processes a query using the query engine and returns the response.

    Args:
        query (str): The query string to be processed by the query engine.

    Returns:
        str: The response generated by the query engine based on the input query.
        
    Example:
        >>> query_function("What is Reverse Engineering?")
        'Reverse engineering is the process of deconstructing an object to understand its design, architecture, and functionality.'
    """
    response = query_engine.query(query)
    return response


iface = gr.Interface(
    fn=query_function,
    inputs=gr.Textbox(label="Query"),
    outputs=gr.Textbox(label="Response")
)

iface.launch()
```

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.

![image](https://github.com/mytechnotalent/pa/blob/main/example.png?raw=true)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
