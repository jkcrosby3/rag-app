# RAG System Web Interface

This web interface provides a user-friendly way to interact with the RAG (Retrieval-Augmented Generation) system. It allows users to ask questions, view AI-generated responses, and explore the supporting documents that inform those responses.

## Features

- **Interactive Query Interface**: Ask questions in natural language
- **Document Exploration**: View the supporting documents used to generate responses
- **Topic Filtering**: Filter results by specific topics
- **Session Management**: Save entire conversation sessions for later reference
- **Response Saving**: Save individual responses to disk
- **Performance Metrics**: View detailed performance metrics for each query
- **Cache Statistics**: Monitor the system's caching performance

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements-ui.txt
   ```

2. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your Anthropic API key: `ANTHROPIC_API_KEY=your_api_key_here`
   
   Alternatively, you can enter your API key directly in the web interface.

3. Launch the web interface:
   ```
   streamlit run app.py
   ```

4. The interface will open in your default web browser at `http://localhost:8501`

## Using the Interface

### Asking Questions

1. Type your question in the text area
2. Click "Submit" to process your query
3. View the response and supporting documents

### Filtering Results

Use the sidebar to:
- Select specific topics to filter results
- Adjust the number of retrieved documents
- Change temperature and token settings

### Saving Data

- **Save Response**: Each response has a "Save Response" button that saves the complete response, including supporting documents, to a text file
- **Save Session**: Click "Save Current Session" in the sidebar to save the entire conversation history to a JSON file
- **New Session**: Start a fresh conversation by clicking "Start New Session"

### Advanced Features

- Toggle response caching on/off
- View cache statistics to monitor system performance
- Adjust advanced parameters like temperature and max tokens

## Saved Data Location

- Individual responses are saved in the `responses/` directory
- Complete sessions are saved in the `sessions/` directory

## Troubleshooting

If you encounter issues:
1. Check that your API key is correctly set
2. Ensure the vector database exists at the expected location
3. Verify that all dependencies are installed
4. Check the console for error messages
