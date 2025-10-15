# Mini RAG Application

This is a Retrieval-Augmented Generation (RAG) system built with Python and Streamlit. It lets users upload documents such as PDF, DOCX, or TXT files. The app then processes these files, stores the content in a way that can be searched, and answers questions based on the document content using a language model.

The app combines search (retrieval) with answer generation. It finds relevant parts from the documents and uses a small AI model to create clear answers. Users can see the sources of the answers for trust.



## Features

- **Document Upload**: Upload one or more PDF, DOCX, or TXT files at once. The app extracts text from them.
- **Text Processing**: Breaks large documents into smaller chunks (pieces of text) to make search easier. You can adjust chunk size and overlap in settings.
- **Searchable Storage**: Converts chunks into vector embeddings (numerical representations) and stores them in FAISS, a fast search library. This acts like a simple database for quick lookup.
- **Retrieval**: Uses semantic search (based on meaning, not just keywords) to find the most relevant chunks for your question. Shows confidence scores (how well a chunk matches, from 0 to 1).
- **Answer Generation**: Feeds the relevant chunks and your question to a local AI model (FLAN-T5-base from Hugging Face). The model generates a concise answer grounded in the document.
- **Citations**: Automatically shows which document and chunk the answer comes from.
- **Configurable Settings**: Adjust chunk size (in words), overlap between chunks, number of retrieved chunks (top K), and minimum confidence score.
- **User Interface**: A clean, wide web app with Streamlit. Everything is centered and uses the full screen width. Settings are in a collapsible section to keep the main screen simple.
- **Session Persistence**: Uploaded documents and processed data stay available during your browser session, even if you ask multiple questions.

The app runs locally on your computer, so no internet is needed after setup (except for initial library downloads).

## Requirements

To run this app, you need:
- Python 3.8 or higher.
- A computer with at least 4 GB RAM (more is better for large documents).
- No GPU is required, but it runs faster with one.

The app uses these Python libraries:
- Streamlit: For the web interface.
- PyPDF2: To read PDF files.
- python-docx: To read DOCX files.
- sentence-transformers: To create embeddings (numerical versions of text for search).
- faiss-cpu: For fast vector search (use faiss-gpu if you have a GPU).
- numpy: For handling numbers and arrays.
- transformers: For the language model (FLAN-T5-base).

## Installation

Follow these steps to set up the app on your computer.

1. **Install Python**: If you do not have Python, download it from python.org. Choose version 3.8 or later. Python is a programming language that runs the code.

2. **Create a Folder for the Project**: Make a new folder on your computer, for example, name it "mini-rag-app". Put the app.py file (the main code) inside this folder.

3. **Open a Terminal or Command Prompt**: This is a text window where you type commands. On Windows, search for "cmd". On Mac, search for "Terminal".

4. **Navigate to the Project Folder**: Type `cd path-to-your-folder` and press Enter. Replace "path-to-your-folder" with the actual location, like `cd Desktop/mini-rag-app`.

5. **Create a Virtual Environment**: This is like a separate box for the app's tools, so they do not mix with other Python stuff on your computer. Type `python -m venv venv` and press Enter. Then activate it:
   - On Windows: `venv\Scripts\activate`
   - On Mac/Linux: `source venv/bin/activate`
   You will see (venv) in the terminal.

6. **Install Dependencies**: These are the libraries the app needs. Type this command and press Enter:
   ```
   pip install streamlit sentence-transformers faiss-cpu numpy transformers torch PyPDF2 python-docx
   ```
   - pip is a tool that downloads and installs libraries.
   - If you have a GPU and want faster performance, use `faiss-gpu` instead of `faiss-cpu`.
   - Torch is needed for the AI models. It may take a few minutes to download.

7. **Verify Installation**: Type `pip list` to see the installed libraries.

If you get errors, check your internet connection or Python version. Search online for the error message if needed.

## How to Run the Application

1. Make sure you are in the project folder and the virtual environment is activated (you see (venv) in the terminal).

2. Type this command and press Enter:
   ```
   streamlit run app.py
   ```
   - Streamlit is the tool that turns the code into a web app.
   - app.py is the main file with all the code.

3. A web browser will open automatically at http://localhost:8501. This is your local server (a mini website on your computer).

4. If it does not open, copy the URL from the terminal and paste it into your browser.

5. To stop the app, go back to the terminal and press Ctrl + C.

The app may take 1-2 minutes to load the AI model the first time.

## Example Usage

Here is a step-by-step guide to use the app. Even if you are new to tech, follow along.

1. **Open the App**: After running, you see the title "Mini RAG (Retrieval-Augmented Generation) Application".

2. **Adjust Settings (Optional)**: Click the arrow next to "Configuration Settings" to open it.
   - Chunk size (words): How big each piece of text is (default 300). Larger chunks hold more info but may slow search.
   - Chunk overlap (words): How much chunks share text (default 50). Helps connect ideas across pieces.
   - Top K retrieved chunks: How many pieces to fetch (default 3).
   - Minimum confidence score: Lowest match quality to show (default 0.2). Higher means stricter results.
   Close the section when done.

3. **Upload a Document**:
   - Under "Upload Documents (PDF, DOCX, or TXT)", click to select files or drag and drop.
   - Example: Upload a file named "Machine Learning Basics.pdf" (you can create a simple PDF with text about machine learning).
   - The app shows "Processing uploaded documents..." then a success message like "X new chunks processed and added. Total chunks available: Y".
   - It then generates embeddings (takes 10-30 seconds). Embeddings are like math codes for text search.

4. **Ask a Question**:
   - Under "Submit Your Query", type in the box: "What is supervised learning, according to the document?"
   - Press Enter.
   - The app shows "Retrieving relevant content..." then retrieved passages (chunks) with document name, chunk number, and confidence score.
   - It then generates the answer using the AI model.
   - Example Output:
     - Retrieved Passages: Shows 1-3 chunks from the PDF.
     - Generated Answer: "Supervised learning involves training a model on a labeled dataset..." (exact text from the document, rephrased clearly).
     - Sources: Machine Learning Basics.pdf - Chunk 1.

5. **Try More**:
   - Upload another file and ask questions. Data accumulates in the session.
   - If no match: It says a polite message like "No relevant content found...".
   - Refresh the browser to start over (clears everything).

For best results:
- Use clear questions.
- Documents should be text-based (no heavy images).
- Large files may take time to process.

## How It Works

- **Ingestion**: When you upload, the app reads the file. For PDF, it goes page by page. For DOCX, paragraph by paragraph. TXT is direct.
- **Chunking**: Splits text into small parts (like paragraphs) to avoid overload.
- **Embeddings**: Turns chunks into vectors (lists of numbers representing meaning) using a pre-trained model.
- **Search**: Your question becomes a vector too. FAISS finds closest matches (like finding similar sentences).
- **Generation**: Puts matches into a prompt (instructions) for the AI model. The model answers only from that info.
- **UI**: Streamlit makes the web page. CSS styles make it look professional and wide.

## Limitations

- Runs on your local machine, so close the terminal to stop.
- In-memory storage: Data clears on refresh or close.
- Small model (FLAN-T5-base): Good for basics, but not as smart as big AI like GPT.
- No save/load: For persistent use, restart and re-upload.
- File size: Limit to under 200 MB per file to avoid slowness.

## Troubleshooting

- **Error on Install**: Update pip with `pip install --upgrade pip`.
- **Model Download Fails**: Check internet. Hugging Face may need login for some models, but not this one.
- **Slow Performance**: Use smaller chunks or fewer documents. Close other apps.
- **No Answer**: Rephrase question or lower confidence threshold.
- **Browser Issues**: Use Chrome or Firefox.

If stuck, search the error in Google or ask on Stack Overflow.
