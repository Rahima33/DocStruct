# DocExtract

DocStruct is a Python tool for extracting, processing, and analyzing documents using computer vision and generative AI models. It leverages Streamlit for interactive UI, OpenCV for image processing, EasyOCR for optical character recognition, and integrates with Google Generative AI and Groq APIs.

## Features
- Document extraction and analysis
- OCR with EasyOCR
# DocStruct

DocStruct is an AI-powered tool that extracts structured data from IT complaint documents, forms, and PDFs. It uses OCR and LLMs to automate field detection and data extraction for helpdesk workflows.
- Generative AI integration (Google, Groq)
Built with Streamlit, OpenCV, EasyOCR, Google Generative AI, Groq, and Roboflow.

## Installation
* Extract fields from scanned IT complaint documents
* OCR with EasyOCR and LLM verification
* Image processing with OpenCV
* Generative AI integration (Google Gemini, Groq)
* Streamlit web interface
* Export results as JSON, CSV, or TXT
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run docextract.py
```

## Configuration


## Dependencies
- streamlit
- numpy
- Pillow
- groq
- easyocr
- python-dotenv
- PyMuPDF
## License

This project is licensed under the MIT License.


Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

