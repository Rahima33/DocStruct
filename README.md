# DocStruct

DocStruct is an AI-powered tool for extracting structured data from IT complaint documents, forms, and PDFs. It uses OCR and LLMs to automate field detection and data extraction for helpdesk workflows.

## Features
* Extract fields from scanned IT complaint documents
* OCR with EasyOCR and LLM verification
* Image processing with OpenCV
* Generative AI integration (Google Gemini, Groq)
* Streamlit web interface
* Export results as JSON, CSV, or TXT

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rahimaej/docstruct.git
   cd docstruct
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

Run the Streamlit app locally:
```
streamlit run main.py
```

## Deployment

Deploy on [Streamlit Cloud](https://streamlit.io/cloud):
1. Push your repo to GitHub.
2. Create a new app on Streamlit Cloud and select `main.py`.
3. Add `.env` and `packages.txt` for API keys and system dependencies.
4. To specify Python version, add `.python-version` with:
   ```
   3.11
   ```

## Configuration

Add your API keys and configuration in the `.env` file:
```
ROBOFLOW_API_KEY=your_key
ROBOFLOW_MODEL_ID=your_model_id
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```

For OpenCV/EasyOCR, add a `packages.txt` file with:
```
libgl1
libglib2.0-0
```

## Dependencies
- streamlit
- numpy
- opencv-python-headless
- pillow
- easyocr
- python-dotenv
- google-genai
- groq
- inference-sdk
- PyMuPDF
- torch
- torchvision

## License

This project is licensed under the MIT License.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or support, open an issue or contact the maintainer.

