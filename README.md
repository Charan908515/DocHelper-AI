# 🧠 DocHelper‑AI

Interactive AI-powered tool to upload documents (PDF, PPT, etc.), chat about their content, generate summaries, search relevant YouTube videos, and download videos, summaries, and chat history.

---

## 🚀 Features

- **Document Upload**: Supports PDF, PPTX, and other text-rich formats.
- **Conversational Chat**: Ask questions and receive AI-generated responses based on your document.
- **Summarization**: Create concise summaries of the uploaded content.
- **YouTube Search Integration**: Automatically find relevant videos related to your document’s content.
- **Downloads**: Save conversation history, summaries, and YouTube videos locally.
- **Web App Interface**: Easy-to-use front-end built with Flask.

---

## 📦 Getting Started

### Requirements

- Python 3.8+
- pip (Python package installer)

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Charan908515/DocHelper-AI.git
   cd DocHelper-AI
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Configure API Keys**  
   If any APIs used (e.g. OpenAI, YouTube), add your credentials as environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export YOUTUBE_API_KEY="your_youtube_key"
   ```

4. **Run the Flask App**  
   ```bash
   python app.py
   ```
   Navigate to http://127.0.0.1:5000 in your web browser.

---

## 🗂️ Project Structure

```
DocHelper‑AI/
├── app.py             # Main Flask app
├── requirements.txt   # Python dependencies
├── templates/         # Flask HTML templates
│   ├── index.html
│   └── result.html
└── README.md          # This file
```

---

## 🛠️ Usage

1. Open the web app in your browser.
2. Upload a PDF, PPT, or supported document.
3. Chat with the uploaded content — ask questions, request summaries, etc.
4. Browse suggested YouTube videos.
5. Download responses, videos, summaries, and chat logs as needed.

---

## ✅ Contributing

Contributions are welcome! Feel free to:

- Report issues or bugs
- Suggest new features
- Submit pull requests

Please follow standard GitHub workflow (fork → branch → commit → pull request).

---

## 💡 Ideas & Improvements

- Support additional document formats (DOCX, TXT, etc.)
- Enable playlist creation of relevant videos
- Allow fine-tuning or customization of summaries
- Integrate login/auth for personalized experiences

---

## 📄 License

This project is released under the **MIT License**.