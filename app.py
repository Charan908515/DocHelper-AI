from flask import Flask, request, jsonify, render_template,send_file,make_response,redirect,url_for,session
from xhtml2pdf import pisa
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory  # Added for Conversational RAG
import os
import secrets
import random
import hashlib
import sqlite3
import markdown
import time
from dotenv import load_dotenv
import nltk
import fitz
from googleapiclient.discovery import build
from PIL import Image
import requests
import yt_dlp
import uuid
from io import BytesIO
from nltk.corpus import stopwords
import smtplib
import bcrypt  # Secure password hashing
from email.mime.text import MIMEText
reset_tokens={}
#nltk.download("stopwords")
#nltk.download("punkt")
load_dotenv()
reset_password={}
os.environ["FLASK_ENV"] = "development"
os.environ["FLASK_DEBUG"] = "1"# Prevent double execution

app = Flask(__name__)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY1"), model_name="llama-3.3-70b-versatile")

db_path = "chat_history.db"
chat_store = {}

### --- DATABASE INITIALIZATION --- ###
def init_db():
    """Initialize the database with sessions and messages tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT CHECK(role IN ('user', 'assistant')), 
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')

    conn.commit()
    conn.close()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL") # Use environment variables
EMAIL_PASSWORD = os.getenv("SMTP_KEY") # Use App Password if 2FA is enabled
# OAuth Configuration  line number --- 86 to 160 ----


# Initialize SQLite Database
def init_user_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Create Users table (Stores hashed passwords)
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')

    # Create OTP table
    cursor.execute('''CREATE TABLE IF NOT EXISTS otps (
                        email TEXT UNIQUE NOT NULL,
                        otp TEXT NOT NULL,
                        timestamp INTEGER NOT NULL)''')

    conn.commit()
    conn.close()
import threading

def delete_empty_sessions():
    """Delete sessions that have no messages and are older than 5 minutes."""
    while True:
        time.sleep(30)  # Check every 30 seconds
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM sessions 
                WHERE session_id IN (
                    SELECT s.session_id
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id
                    WHERE m.session_id IS NULL
                    AND s.created_at <= datetime('now', '-5 minutes')
                )
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error deleting empty sessions: {e}")
        finally:
            conn.close()


# Run delete_empty_sessions in a background thread
threading.Thread(target=delete_empty_sessions, daemon=True).start()
def generate_otp_email_html(otp):
    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f4f4;
                padding: 20px;
            }}
            .container {{
                max-width: 500px;
                margin: auto;
                background-color: #ffffff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            .title {{
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .otp {{
                font-size: 32px;
                color: #2980b9;
                font-weight: bold;
                margin: 20px 0;
            }}
            .note {{
                color: #7f8c8d;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">🔐 Your OTP Code</div>
            <p>Hello,</p>
            <p>Please use the OTP below to complete your verification:</p>
            <div class="otp">{otp}</div>
            <p class="note">This code is valid for a limited time. Do not share this with anyone.</p>
            <p>Thank you,<br><strong>Team Doc Helper AI</strong></p>
        </div>
    </body>
    </html>
    """


def send_email(receiver_email, subject, message):
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure connection
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  

        msg = MIMEText(message,"html")
        msg["Subject"] = subject
        msg["From"] = "Login OTP for DocHelper AI"
        msg["To"] = receiver_email

        server.sendmail(EMAIL_ADDRESS, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Error sending email:", str(e))
        return False

### --- SESSION MANAGEMENT --- ###
    
def generate_session_id():
    """Generate a unique session ID"""
    unique_str = f"{time.time()}-{secrets.token_hex(8)}"
    return hashlib.sha256(unique_str.encode()).hexdigest()

def create_session(user_id,session_name=None):
    """Create a new session for the user."""
    session_id = generate_session_id()
    if not session_name:
        session_name = f"Session {time.strftime('%Y-%m-%d %H:%M:%S')}"  # Default name format

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO sessions (session_id, user_id,session_name) VALUES (?,?, ?)", (session_id, user_id,session_name))
    
    conn.commit()
    conn.close()
    
    return session_id,session_name


### --- CHAT HISTORY STORAGE --- ###
def save_message(session_id, user_id, role, message):
    """Save a chat message in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO messages (session_id, user_id, role, message) VALUES (?, ?, ?, ?)",
                   (session_id, user_id, role, message))
    
    conn.commit()
    conn.close()




### --- API ROUTES --- ###

@app.route('/start_session', methods=['POST'])
def start_session():
    """API to start a new session for a user."""
    data = request.json
    user_id = data.get("user_id")
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    session_id = create_session(user_id)
    return jsonify({"session_id": session_id})

@app.route('/')
def index():
    return render_template('landingpage.html')
@app.route("/login")
def login_page():
    return render_template("login.html")  # Ensure login.html is in templates folder
@app.route("/register_page")
def register_page():
    return render_template("registration.html")
@app.route("/index")
def home_page():
    return render_template("index-1.html")  # Ensure registration.html is in templates folder
@app.route("/profile")
def profile_page():
     return render_template("profile.html")
@app.route("/summary_page")
def summary_page():
    return render_template("summary.html")
@app.route("/history")
def history_page():
    return render_template("history.html")
# Send OTP and Store in Database
@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.json
    email = data.get("email")
    
    if not email:
        return jsonify({"message": "Email is required"}), 400

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Check if user already exists
    #cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    #if cursor.fetchone():
    #    conn.close()
    #    return jsonify({"success": False, "message": "User already exists. Redirecting to login.", "redirect": "login.html"}), 200

    # Generate and store OTP
    otp = str(random.randint(100000, 999999))
    html_message = generate_otp_email_html(otp)
    timestamp = int(time.time())

    cursor.execute("DELETE FROM otps WHERE email = ?", (email,))  # Ensure only one OTP per email
    cursor.execute("INSERT INTO otps (email, otp, timestamp) VALUES (?, ?, ?)", (email, otp, timestamp))
    conn.commit()
    conn.close()

    # Send OTP via email
    if send_email(email, "Your OTP Code",html_message):
        return jsonify({"success":True,"message": "OTP sent successfully!"}), 200
    else:
        return jsonify({"sucess":False,"message": "Failed to send OTP"}), 500

# Verify OTP from Database
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")

    if not email or not otp:
        return jsonify({"message": "Email and OTP are required"}), 400

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT otp FROM otps WHERE email = ?", (email,))
    stored_otp = cursor.fetchone()

    if stored_otp and stored_otp[0] == otp:
        cursor.execute("DELETE FROM otps WHERE email = ?", (email,))  # OTP should be used only once
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "OTP verified successfully!", "redirect": "/index"}), 200
    else:
        conn.close()
        return jsonify({"success": False, "message": "Invalid OTP"}), 400

# Register User
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name=data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not email:
        return jsonify({"message": "Email is required"}), 400
    if not name:
        return jsonify({"message": "username is required"}), 400
    if not password:
        return jsonify({"message": "Password are required"}), 400
    
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO users (email,name, password) VALUES (?,?, ?)", (email,name, hashed_password.decode('utf-8')))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Registration successful", "redirect": "/index.html"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"message": "User already registered"}), 400

# Login User
@app.route('/user-login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user[0].encode()):
        return jsonify({"success": True, "message": "Login successful!"}), 200
    else:
        return jsonify({"success": False, "message": "Invalid email or password"}), 401


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        
        # Check if email exists
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if user:
            token = str(uuid.uuid4())
            reset_tokens[token] = email  # Save token
            reset_link = f"http://localhost:5000/reset-password/{token}"
            body = f"""
                        <html>
                        <body style="font-family: Arial, sans-serif; background-color: #ffffff; padding: 20px;">
                            <div style="max-width: 600px; margin: auto; border: 1px solid #ddd; border-radius: 10px; padding: 30px;">
                                <h2 style="text-align: center; color: #333;">Forgotten your password?</h2>
                                <p>Hi there,</p>
                                <p>Forgotten your password for <strong>Doc Helper AI</strong>? Don’t worry, resetting it couldn’t be easier. Just click the button below.</p>

                                <div style="text-align: center; margin: 30px 0;">
                                    <a href="{reset_link}" style="background-color: #f6a623; color: white; padding: 14px 28px; text-decoration: none; font-weight: bold; border-radius: 5px;">Reset Password</a>
                                </div>

                                    <p style="font-size: 0.9em; color: #666;">
                                        For security purposes, this link will expire in 30 minutes or after you reset your password. If you didn’t request a password reset, please ignore this message.
                                    </p>

                                    <p style="font-size: 0.9em; color: #666;">
                                        If the button above doesn’t work, paste this link into your browser:<br>
                                        <a href="{reset_link}" style="color: #1a0dab;">{reset_link}</a>
                                    </p>

                                    <p style="margin-top: 40px;">With best regards,<br><strong>Doc Helper AI Team</strong></p>
                                    </div>
                                </body>
                                </html>
                                """
            send_email(email,"Password Reset Code",body)
            return """<h1 style="'text-align':'center','margin':100 500 600 600"> A reset link was sent to your email."</h1>"""
    return render_template("forgot_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    email = reset_tokens.get(token)
    if not email:
        return "Invalid or expired token", 400

    if request.method == "POST":
        new_password = request.form["new_password"]
        hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())

        # Update password in DB
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
        conn.commit()
        conn.close()

        reset_tokens.pop(token, None)  # Invalidate token
        return redirect("/login")
    
    return render_template("reset_password.html", email=email)



@app.route('/upload', methods=['POST'])
def upload_pdf():
    user_id = request.form.get("email")
    session_name=request.form.get("session_name")
    print("upload:",user_id)
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    documents = []
    file_paths = []  # Store file paths

    for file in files:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        file_paths.append(file_path)  # Save path
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    session_id ,session_name= create_session(user_id,session_name)
    print(session_id)
    if not splits:
        return jsonify({"error": "No readable text found in the uploaded file(s). Please upload a valid text-based PDF."}), 400
    if user_id not in chat_store:
        chat_store[user_id] = {}

    chat_store[user_id][session_id] = {
        "retriever": retriever,
        "history": ChatMessageHistory(),
        "file_paths": file_paths,  # Store file paths
        "empty":True
    }
    
    return jsonify({
        "message": "Files uploaded successfully.",
        "session_id": session_id,
        "session_name":session_name
    })

def get_uploaded_file_paths(user_id, session_id):
    """Retrieve uploaded file paths for a session."""
    if user_id in chat_store and session_id in chat_store[user_id]:
        return chat_store[user_id][session_id].get("file_paths", [])
    return []

def get_session_history(session_id, user_id):
    return chat_store.get(user_id, {}).get(session_id, {}).get("history", ChatMessageHistory())


@app.route('/chat', methods=['POST'])
def chat():
    print("chat activated")
    data = request.get_json()
    user_id = data.get("user_id")
    session_id = data.get("session_id")
    question = data.get("question", "").strip()
    print(user_id)
    if not user_id or not session_id or not question:
        return jsonify({"error": "User ID, Session ID, and Question are required"}), 400

    
    if user_id not in chat_store or session_id not in chat_store[user_id]:
        return jsonify({"error": "Invalid session. Please upload documents again."}), 400

    retriever = chat_store[user_id][session_id]["retriever"]
    history = chat_store[user_id][session_id]["history"]

    chat_history_messages = history.messages if history else []
    chat_store[user_id][session_id]["empty"] = False
    try:
        
        retrieved_docs = retriever.invoke(question)
        print(retrieved_docs)

        if not retrieved_docs:
            answer = "I don’t know. The provided documents do not contain relevant information."
            return jsonify({"answer": answer})

        retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

        
        system_prompt = (
            "You are an AI assistant answering questions based on retrieved context. "
            "Use  the provided context to answer the question concisely and shortly"
            "If the context does not contain an answer, use history and answer your own words concisely and shortly. " 
            "\n\nContext:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
       
        print("rag chain:",rag_chain)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: get_session_history(session_id, user_id),  # Ensure user_id is passed
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        print("response:",response)
        answer =response["answer"]
        answer=markdown.markdown(answer)

        save_message(session_id,user_id,"user",question)
        save_message(session_id,user_id,"assistant",answer)

        # Append to chat history
        history.add_message({"role": "user", "content": question})
        history.add_message({"role": "assistant", "content": answer})


        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error in chat function: {e}")
        return jsonify({"error": "Error processing request. Please try again."}), 500


def generate_summary(text):
    """Generate a structured summary using an LLM."""
    if not text:
        return "No text available for summarization."

    
    model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY2"))
    parser = StrOutputParser()

    
    prompt = PromptTemplate(
        template=(
            "Summarize the following stopwords-removed text in a structured format:\n\n"
            "### Key Points:\n"
            "- Provide clear headings for each topic.\n"
            "- Keep explanations concise yet detailed.\n"
            "- Use bullet points for important details.\n\n"
            "Text:\n{data}"
        ),
        input_variables=["data"]
    )

    chain = prompt | model | parser
    summary = chain.invoke({"data": text})

    return summary


@app.route("/summary", methods=["POST"])
def fetch_summary():
    print("summary activated")
    """Main API endpoint to generate a summary."""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        session_id = data.get("session_id")

        if not user_id or not session_id:
            return jsonify({"error": "User ID and Session ID are required."}), 400
        content=""
        
        file_paths=get_uploaded_file_paths(user_id, session_id)
        for path in file_paths:
            with fitz.open(path) as doc:
                for page in doc:
                    content += page.get_text("text") + "\n"
        print("file path:",file_paths)
         # Tokenize words and remove stopwords
        words =content.split()
        filtered_content = " ".join([word for word in words if word.lower() not in set(stopwords.words("english"))])
        
        summary = generate_summary(filtered_content)
        summary=markdown.markdown(summary)
        print(summary)
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error in /youtube function: {e}")
        return jsonify({"error": "Error processing request. Please try again."}), 500


def get_download_link(video_url):
    ydl_opts = {
        "format": "best",
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info.get("url", None)
    
def youtube_search(query, language,duration,max_results=2):
    api_key =os.getenv("cloud_console_api_key2")
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.search().list(
        part='snippet',
        q=query,
        type='video',
         relevanceLanguage=language,  # Set the language filter (e.g., "en" for English, "hi" for Hindi, "te" for telugu)
         videoDuration=duration,
        maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response.get('items', []):
        video_data = {
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'channel_title': item['snippet']['channelTitle'],
            'publish_time': item['snippet']['publishTime'],
            'video_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
            "download_link": get_download_link(f"https://www.youtube.com/watch?v={item['id']['videoId']}")
        }
        videos.append(video_data)

    return videos

@app.route("/redirect_youtube_videos")
def redirect_youtube():
    return render_template("you_video.html")


@app.route("/fetch_youtube_videos", methods=["POST"])
def fetch_youtube_videos():
    print("youtube fetching activated")
    """Another function that calls youtube_videos logic internally."""
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        session_id = data.get("session_id")

        if not user_id or not session_id:
            return jsonify({"error": "User ID and Session ID are required."}), 400
        content=""
        file_paths=get_uploaded_file_paths(user_id, session_id)
        for path in file_paths:
            with fitz.open(path) as doc:
                for page in doc:
                    content += page.get_text("text") + "\n"
        
         # Tokenize words and remove stopwords
        words =content.split()
        filtered_content = " ".join([word for word in words if word.lower() not in set(stopwords.words("english"))])
        
        summary = generate_summary(filtered_content)
        topics_prompt=PromptTemplate(
        template="""give the important topic names from the summarized content to search to youtube like
                    [topic1,topic2,....] do not give anything else.The summarized content:{content}""",
        input_variables=["content"],
        )
        parser=StrOutputParser()
        topic_model=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=os.getenv("GROQ_API_KEY3")) # Changed 'moel' to 'model'
        topics_chain=topics_prompt|topic_model|parser
        topics=topics_chain.invoke({"content":summary})
        topics=topics.split(",")
        topics=[topic.strip() for topic in topics]
        video_results=[]
        for topic in topics:
            results = youtube_search(query=topic,language="te",duration="medium")

            for video in results:
                video_results.append({
                    "title": video["title"],
                    "description": video["description"],
                    "video_url": video["video_url"],
                    "thumbnail_url": video["thumbnail_url"],
                    "download_link": video["download_link"]
                })

        return jsonify({"videos":video_results})

    except Exception as e:
        print(f"Error in /process_summary function: {e}")
        return jsonify({"error": "Error processing summary. Please try again."}), 500

@app.route("/get_user_sessions", methods=["GET"])
def get_user_sessions():
    """Retrieve all sessions and their chat history for a given user."""
    user_id = request.args.get("email")

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get all sessions for the user
        cursor.execute("SELECT session_id, session_name FROM sessions WHERE user_id = ?", (user_id,))
        sessions_data = cursor.fetchall()

        sessions = {}
        for session_id, session_name in sessions_data:
            # Check if the session has any messages
            cursor.execute(
                "SELECT role, message FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            messages = cursor.fetchall()

            if messages:  # Only include sessions that have chat messages
                sessions[session_id] = {
                    "session_name": session_name,
                    "history": [{"role": role, "message": message} for role, message in messages]
                }

    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

    return jsonify({"sessions": sessions})



@app.route("/download_summary_pdf", methods=["POST"])
def download_summary_pdf():
    data = request.get_json()
    summary_html = data.get("summary")

    if not summary_html:
        return jsonify({"error": "No summary content provided"}), 400

    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(summary_html, dest=pdf)

    if pisa_status.err:
        return jsonify({"error": "Failed to generate PDF"}), 500

    pdf.seek(0)
    response = make_response(pdf.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=summary.pdf'
    return response
@app.route("/change_password", methods=["POST"])
def change_password():
    data = request.get_json()
    email = data.get("email")
    current_password = data.get("old_password")
    new_password = data.get("new_password")

    if not email or not current_password or not new_password:
        return jsonify({"success": False, "message": "All fields are required."}), 400

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()

        if not result:
            return jsonify({"success": False, "message": "User not found."}), 404

        hashed_pw = result[0]
        if not bcrypt.checkpw(current_password.encode('utf-8'), hashed_pw.encode('utf-8')):
            return jsonify({"success": False, "message": "Current password is incorrect."}), 401

        new_hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("UPDATE users SET password = ? WHERE email = ?", (new_hashed_pw.decode('utf-8'), email))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Password changed successfully."}), 200

    except Exception as e:
        print(f"Error in /change_password: {e}")
        return jsonify({"success": False, "message": "Internal server error."}), 500


@app.route("/rename_session", methods=["POST"])
def rename_session():
    data = request.get_json()
    session_id = data.get("session_id")
    new_name = data.get("session_name")

    if not session_id or not new_name:
        return jsonify({"error": "Missing data"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET session_name = ? WHERE session_id = ?", (new_name, session_id))
    conn.commit()
    conn.close()
    return jsonify({"message": "Session renamed successfully"})
   

@app.route("/delete_session", methods=["POST"])
def delete_session():
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": "Session deleted"})


@app.route("/export_session_pdf", methods=["POST"])
def export_session_pdf():
    data = request.get_json()
    user_id = data.get("user_id")
    session_id = data.get("session_id")

    if not user_id or not session_id:
        return jsonify({"error": "User ID and session ID are required"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
    messages = cursor.fetchall()
    conn.close()

    chat_html = "<h2>Chat Session Export</h2><hr>"
    for role, message, timestamp in messages:
        style = "color:blue;" if role == "user" else "color:green;"
        chat_html += f"<p><strong style='{style}'>{role.title()}:</strong> {message}</p>"

    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(chat_html, dest=pdf)
    if pisa_status.err:
        return jsonify({"error": "Failed to generate PDF"}), 500

    pdf.seek(0)
    return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name="chat_session.pdf")


@app.route("/pdf/<session_id>", methods=["GET"])
def get_pdf_by_session(session_id):
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    file_paths = chat_store.get(user_id, {}).get(session_id, {}).get("file_paths", [])
    if not file_paths:
        return jsonify({"error": "No PDF found for this session"}), 404


 
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    init_user_db()
    init_db()

    app.run(host="localhost", port=4200,debug=False)  # Set debug=True for development
    # Run delete_empty_sessions in a background thread
    

    
