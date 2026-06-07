# RAG Web Application TODO List

## High Priority - UI/UX Improvements

### 1. Chat Interface Improvements
- [ ] **Scrollable Chat Container**
  - Create fixed-height container with scrollbar for chat history
  - Position query input box at bottom (fixed)
  - Display answers above the input box
  - Auto-scroll to latest answer after each submission
  - Similar to ChatGPT interface pattern

- [ ] **Chat History Management**
  - [ ] Save conversations to disk
    - Format options: .txt, .md, or .csv
    - Include timestamps for each query/response
    - Save location: `data/chat_history/` or user-specified directory
  - [ ] Add "Export Chat" button
    - Allow user to choose format (.txt, .md, .csv)
    - Include metadata (date, time, number of documents retrieved)
  - [ ] Add "Clear Chat" button with confirmation
  - [ ] Add "Load Previous Chat" functionality

### 2. Query Interface Layout
- [ ] Redesign query tab layout:
  ```
  ┌─────────────────────────────────────┐
  │  Chat History (scrollable)          │
  │  ┌───────────────────────────────┐  │
  │  │ Q: Previous question          │  │
  │  │ A: Previous answer...         │  │
  │  │                               │  │
  │  │ Q: Another question           │  │
  │  │ A: Another answer...          │  │
  │  │                               │  │
  │  │ [Auto-scroll to here]         │  │
  │  └───────────────────────────────┘  │
  │                                     │
  │  ┌───────────────────────────────┐  │
  │  │ Enter your question...        │  │
  │  │ [Submit]                      │  │
  │  └───────────────────────────────┘  │
  └─────────────────────────────────────┘
  ```

## Medium Priority - Features

### 3. Enhanced Chat Features
- [ ] Add "Copy Answer" button for each response
- [ ] Add "Regenerate" button to re-run last query
- [ ] Show loading indicator during query processing
- [ ] Display token count and cost estimate per query
- [ ] Add "Sources" toggle to show/hide retrieved documents

### 4. Session Management
- [ ] Auto-save chat every N messages
- [ ] Session naming/tagging
- [ ] Search through previous sessions
- [ ] Session statistics (total queries, avg response time)

### 5. Export Formats

#### Text Format (.txt)
```
Revolutionary War RAG Chat History
Date: 2026-05-08
Time: 9:58 PM
========================================

[9:58 PM] User: how many soldiers are in the pension files?
[9:58 PM] Assistant: The database contains 12,606 Revolutionary War pension files...
Documents Retrieved: 20
Response Time: 3.2s

[10:01 PM] User: tell me about George Washington
[10:01 PM] Assistant: Based on the documents...
Documents Retrieved: 15
Response Time: 2.8s
```

#### Markdown Format (.md)
```markdown
# Revolutionary War RAG Chat History

**Date:** 2026-05-08  
**Time:** 9:58 PM

---

## Query 1 (9:58 PM)
**User:** how many soldiers are in the pension files?

**Assistant:** The database contains 12,606 Revolutionary War pension files...

**Metadata:**
- Documents Retrieved: 20
- Response Time: 3.2s

---

## Query 2 (10:01 PM)
...
```

#### CSV Format (.csv)
```csv
timestamp,user_query,assistant_response,documents_retrieved,response_time_seconds
2026-05-08 21:58:00,"how many soldiers are in the pension files?","The database contains 12,606...",20,3.2
2026-05-08 22:01:00,"tell me about George Washington","Based on the documents...",15,2.8
```

## Low Priority - Polish

### 6. Visual Improvements
- [ ] Add alternating background colors for Q&A pairs
- [ ] Add user/assistant avatars
- [ ] Improve markdown rendering in responses
- [ ] Add syntax highlighting for code blocks
- [ ] Better mobile responsiveness

### 7. Settings & Preferences
- [ ] User preference for auto-save location
- [ ] User preference for export format
- [ ] Theme selection (light/dark mode)
- [ ] Font size adjustment

## Technical Implementation Notes

### Streamlit Chat Container
```python
# Use st.container with custom CSS for scrollable area
chat_container = st.container()
with chat_container:
    # Display chat history
    for msg in st.session_state.chat_history:
        # Render Q&A pairs
        pass

# Fixed input at bottom
st.text_input("Your question:", key="query_input")
```

### Auto-scroll JavaScript
```python
# Inject JavaScript to scroll to bottom
st.markdown("""
<script>
    var chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
</script>
""", unsafe_allow_html=True)
```

### Save Chat Function
```python
def save_chat_history(format='txt', filepath=None):
    if filepath is None:
        filepath = f"data/chat_history/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    
    if format == 'txt':
        save_as_text(st.session_state.chat_history, filepath)
    elif format == 'md':
        save_as_markdown(st.session_state.chat_history, filepath)
    elif format == 'csv':
        save_as_csv(st.session_state.chat_history, filepath)
```

## Dependencies Needed
- None (all features can be implemented with existing Streamlit)
- Consider `streamlit-chat` component for better chat UI (optional)

## Files to Modify
- `src/web/app.py` - Main UI changes
- `src/utils/chat_export.py` - New file for export functionality
- `src/web/styles.css` - New file for custom CSS (optional)

## Testing Checklist
- [ ] Chat scrolls correctly on new messages
- [ ] Export works for all formats (.txt, .md, .csv)
- [ ] Chat history persists during session
- [ ] Auto-save doesn't interfere with user experience
- [ ] Mobile layout is usable
- [ ] Large chat histories don't cause performance issues

## Future Enhancements (Post-Hackathon)
- [ ] Multi-user support with separate chat histories
- [ ] Share chat via link
- [ ] Annotate/highlight important responses
- [ ] Export to PDF
- [ ] Voice input/output
- [ ] Integration with external note-taking apps

---

**Priority Legend:**
- High Priority: Essential for good user experience
- Medium Priority: Nice to have, improves usability
- Low Priority: Polish and extra features

**Created:** 2026-05-08  
**Last Updated:** 2026-05-08
