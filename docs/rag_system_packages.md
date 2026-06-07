# RAG System Python Packages and Functions

## 1. Startup and Initialization

### Core Python Packages:
- `os`: File system operations
  - `os.path.join()`: Path manipulation
  - `os.getenv()`: Environment variable access

- `sys`: System operations
  - `sys.path`: Python path management

- `datetime`: Date and time handling
  - `datetime.datetime.now()`: Current time
  - `datetime.datetime.strftime()`: Date formatting

- `pathlib`: Path operations
  - `Path()`: Path manipulation
  - `Path.parent`: Directory navigation

### Third-party Packages:
- `python-dotenv`: Environment variable management
  - `load_dotenv()`: Load environment variables

- `logging`: Logging framework
  - `logging.basicConfig()`: Configure logging
  - `logging.getLogger()`: Get logger instance

## 2. Web Interface

### Web Framework:
- `flask`: Web server framework
  - `Flask()`: Create application
  - `render_template()`: Template rendering
  - `request`: HTTP request handling
  - `jsonify()`: JSON response creation
  - `session`: Session management

### Data Processing:
- `json`: JSON handling
  - `json.loads()`: Parse JSON
  - `json.dumps()`: Create JSON

- `pandas`: Data analysis
  - `pd.DataFrame()`: Data manipulation
  - `pd.read_csv()`: CSV reading

## 3. Query Processing

### Vector Operations:
- `faiss`: Vector similarity search
  - `faiss.IndexFlatIP`: Inner product index
  - `faiss.IndexFlatL2`: L2 distance index
  - `faiss.write_index()`: Save index
  - `faiss.read_index()`: Load index

### Embeddings:
- `sentence-transformers`: Text embeddings
  - `SentenceTransformer()`: Model loading
  - `encode()`: Text embedding generation

### LLM Integration:
- `anthropic`: Claude API
  - `Anthropic()`: API client
  - `complete()`: Text completion

## 4. Document Processing

### PDF Processing:
- `PyMuPDF`: PDF handling
  - `fitz.open()`: Open PDF
  - `page.get_text()`: Extract text
  - `page.get_text_words()`: Extract words

- `PyPDF2`: PDF utilities
  - `PdfReader()`: Read PDF
  - `PdfWriter()`: Write PDF

### Text Processing:
- `nltk`: Natural language processing
  - `word_tokenize()`: Tokenization
  - `sent_tokenize()`: Sentence splitting

## 5. Cache Management

### Caching:
- `functools`: Cache decorators
  - `lru_cache()`: Least Recently Used caching

- `sqlite3`: SQLite database
  - `connect()`: Database connection
  - `execute()`: SQL execution

### Performance Monitoring:
- `time`: Time measurement
  - `time.time()`: Current time
  - `time.perf_counter()`: Performance counter

## 6. Session Management

### Session Handling:
- `flask-session`: Session management
  - `session.get()`: Get session data
  - `session.set()`: Set session data
  - `session.clear()`: Clear session

### State Management:
- `pickle`: Object serialization
  - `pickle.dumps()`: Serialize objects
  - `pickle.loads()`: Deserialize objects

## 7. Testing and Development

### Testing:
- `pytest`: Testing framework
  - `pytest.mark.parametrize()`: Parameterized tests
  - `pytest.fixture()`: Test fixtures

### Development Tools:
- `jupyter`: Interactive development
  - `notebook`: Jupyter notebook interface

### CLI Tools:
- `typer`: Command-line interface
  - `@app.command()`: Command decorators

## 8. Data Visualization

### Plotting:
- `plotly`: Interactive plots
  - `go.Figure()`: Create figures
  - `make_subplots()`: Create subplots

### Data Analysis:
- `numpy`: Numerical operations
  - `np.array()`: Create arrays
  - `np.mean()`: Calculate means

## 9. File Operations

### File Handling:
- `shutil`: File operations
  - `shutil.copy()`: Copy files
  - `shutil.rmtree()`: Remove directories

### Path Operations:
- `pathlib.Path`: Path manipulation
  - `exists()`: Check existence
  - `mkdir()`: Create directories
  - `glob()`: Pattern matching
