"""
Flask web interface for document tracking system.
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from .models import Document, Version, Notification, Base, Tag, TagSynonym
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from werkzeug.utils import secure_filename
import uuid
from .config.llm_config import get_all_llms, get_llm_config
from .config.query_validator import QUERY_VALIDATION
from .tools.tag_validator import TagValidator

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///document_tracking.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# Initialize database
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

tag_validator = TagValidator()

@app.route('/')
def index():
    """Show main interface with document upload and LLM query options."""
    return render_template('main.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document upload."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file:
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(filepath)
            
            # Process the document
            try:
                from .document_processor import process_document
                process_document(
                    filepath,
                    output_dir="processed",
                    metadata={
                        "title": filename,
                        "source": "user_upload",
                        "classification": "public"  # Default classification
                    }
                )
                
                # Add to database
                session = Session()
                document = Document(
                    path=filepath,
                    title=filename,
                    classification="public"
                )
                session.add(document)
                session.commit()
                
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    return render_template('upload.html')

@app.route('/document/<int:doc_id>')
def document_detail(doc_id):
    """Show document details and version history."""
    session = Session()
    document = session.query(Document).filter_by(id=doc_id).first()
    if not document:
        return "Document not found", 404
    
    versions = session.query(Version).filter_by(document_id=doc_id).order_by(Version.version_number.desc()).all()
    notifications = session.query(Notification).filter_by(document_id=doc_id).order_by(Notification.timestamp.desc()).all()
    
    return render_template('document_detail.html', 
                         document=document,
                         versions=versions,
                         notifications=notifications)

@app.route('/version/<int:version_id>')
def version_detail(version_id):
    """Show version details."""
    session = Session()
    version = session.query(Version).filter_by(id=version_id).first()
    if not version:
        return "Version not found", 404
    
    return render_template('version_detail.html', version=version)

@app.route('/revert/<int:doc_id>/<int:version_id>', methods=['POST'])
def revert_version(doc_id, version_id):
    """Revert document to a specific version."""
    session = Session()
    document = session.query(Document).filter_by(id=doc_id).first()
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    target_version = session.query(Version).filter_by(id=version_id).first()
    if not target_version:
        return jsonify({'error': 'Version not found'}), 404
    
    # Create new version entry
    new_version = Version(
        document_id=doc_id,
        version_number=len(document.versions) + 1,
        timestamp=datetime.utcnow(),
        file_size=target_version.file_size,
        file_hash=target_version.file_hash,
        metadata=target_version.metadata,
        changes=json.dumps({"reverted_from": version_id}),
        reverted_from=version_id
    )
    
    # Update document
    document.current_version_id = new_version.id
    document.updated_at = datetime.utcnow()
    
    # Add notification
    notification = Notification(
        event_type="document_revert",
        message=f"Document {document.path} reverted to version {version_id}",
        document_id=doc_id,
        version_id=new_version.id
    )
    
    session.add(new_version)
    session.add(notification)
    session.commit()
    
    return jsonify({'success': True, 'version': new_version.version_number})

@app.route('/notifications')
def notifications():
    """Show all notifications."""
    session = Session()
    notifications = session.query(Notification).order_by(Notification.timestamp.desc()).all()
    return render_template('notifications.html', notifications=notifications)

@app.route('/api/document/<int:doc_id>')
def get_document(doc_id):
    """Get document details via API."""
    session = Session()
    document = session.query(Document).filter_by(id=doc_id).first()
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify({
        'id': document.id,
        'path': document.path,
        'title': document.title,
        'classification': document.classification,
        'versions': len(document.versions),
        'last_updated': document.updated_at.isoformat()
    })

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List documents with filtering."""
    session = Session()
    # Get filters from request
    search = request.args.get('search', '').strip()
    classification = request.args.get('classification', '')
    version = request.args.get('version', '')
    size = request.args.get('size', '')
    upload_date = request.args.get('upload_date', '')
    tag_ids = request.args.getlist('tag_ids')

    query = Document.query

    # Apply search filter
    if search:
        query = query.filter(
            or_(
                Document.filename.ilike(f'%{search}%'),
                Document.description.ilike(f'%{search}%'),
                Document.keywords.ilike(f'%{search}%')
            )
        )

    # Apply classification filter
    if classification:
        query = query.filter(Document.classification == classification)

    # Apply version filter
    if version == 'latest':
        query = query.group_by(Document.filename).having(
            Document.version == func.max(Document.version)
        )
    elif version == 'all':
        # No filtering needed
        pass

    # Apply size filter
    if size:
        if size == 'small':
            query = query.filter(Document.size < 1024 * 1024)  # < 1MB
        elif size == 'medium':
            query = query.filter(
                and_(
                    Document.size >= 1024 * 1024,  # >= 1MB
                    Document.size <= 1024 * 1024 * 10  # <= 10MB
                )
            )
        elif size == 'large':
            query = query.filter(Document.size > 1024 * 1024 * 10)  # > 10MB

    # Apply upload date filter
    if upload_date:
        now = datetime.utcnow()
        if upload_date == 'week':
            one_week_ago = now - datetime.timedelta(days=7)
            query = query.filter(Document.uploaded_at >= one_week_ago)
        elif upload_date == 'month':
            one_month_ago = now - datetime.timedelta(days=30)
            query = query.filter(Document.uploaded_at >= one_month_ago)
        elif upload_date == 'year':
            one_year_ago = now - datetime.timedelta(days=365)
            query = query.filter(Document.uploaded_at >= one_year_ago)

    # Apply tag filter
    if tag_ids:
        query = query.join(DocumentTag).filter(DocumentTag.tag_id.in_(tag_ids))

    # Order by uploaded date (newest first)
    query = query.order_by(Document.uploaded_at.desc())

    documents = query.all()
    return jsonify([doc.to_dict() for doc in documents])

@app.route('/api/tags', methods=['GET'])
def list_tags():
    include_children = request.args.get('include_children', 'false').lower() == 'true'
    include_synonyms = request.args.get('include_synonyms', 'false').lower() == 'true'
    include_stats = request.args.get('include_stats', 'false').lower() == 'true'
    
    tags = Tag.query.all()
    
    if include_stats:
        # Calculate tag usage statistics
        tag_usage = db.session.query(
            DocumentTag.tag_id,
            func.count(DocumentTag.document_id).label('usage_count')
        ).group_by(DocumentTag.tag_id).all()
        
        # Update usage counts
        for tag in tags:
            tag.usage_count = next((u.usage_count for u in tag_usage if u.tag_id == tag.id), 0)
    
    return jsonify([tag.to_dict(include_children, include_synonyms) for tag in tags])

@app.route('/api/tags/suggest', methods=['POST'])
def suggest_tags():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify([])
        
    # Get existing tags
    tags = Tag.query.all()
    
    # Get synonyms
    synonyms = TagSynonym.query.all()
    synonym_map = {syn.synonym.lower(): syn.tag_id for syn in synonyms}
    
    # Extract potential tags from text
    words = set(word.lower() for word in re.findall(r'\w+', text))
    
    # Find matching tags and their synonyms
    suggestions = []
    for word in words:
        # Check if word matches a tag name
        tag = next((t for t in tags if t.name.lower() == word), None)
        if tag:
            suggestions.append({
                'id': tag.id,
                'name': tag.name,
                'score': 1.0,  # Perfect match
                'type': 'tag'
            })
            continue
            
        # Check if word matches a synonym
        tag_id = synonym_map.get(word)
        if tag_id:
            tag = next((t for t in tags if t.id == tag_id), None)
            if tag:
                suggestions.append({
                    'id': tag.id,
                    'name': tag.name,
                    'score': 0.8,  # Synonym match
                    'type': 'synonym'
                })
                continue
                
        # Check for partial matches
        for tag in tags:
            if word in tag.name.lower():
                suggestions.append({
                    'id': tag.id,
                    'name': tag.name,
                    'score': 0.5,  # Partial match
                    'type': 'partial'
                })
                
    # Sort suggestions by score
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    
    return jsonify(suggestions[:10])  # Return top 10 suggestions

@app.route('/api/tags', methods=['POST'])
def create_tag():
    """Create new tag."""
    data = request.get_json()
    
    # Validate the tag data
    errors = tag_validator.validate_tag(data, Tag.query.all())
    if errors:
        return jsonify({'errors': errors}), 400
            
    if not data.get('name'):
        return jsonify({'error': 'Tag name is required'}), 400
            
    parent_id = data.get('parent_id')
    synonyms = data.get('synonyms', [])
    
    tag = Tag(
        name=data['name'],
        color=data.get('color'),
        description=data.get('description'),
        parent_id=parent_id
    )
    
    try:
        db.session.add(tag)
        db.session.flush()  # Get the tag ID
        
        # Create synonyms
        for synonym in synonyms:
            db.session.add(TagSynonym(
                tag_id=tag.id,
                synonym=synonym
            ))
        
        db.session.commit()
        return jsonify(tag.to_dict(True, True))
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags/<int:tag_id>', methods=['PUT'])
def update_tag(tag_id):
    tag = Tag.query.get_or_404(tag_id)
    data = request.get_json()
    
    # Validate the tag data
    errors = tag_validator.validate_tag(data, Tag.query.all(), tag_id)
    if errors:
        return jsonify({'errors': errors}), 400
            
    tag.name = data.get('name', tag.name)
    tag.color = data.get('color', tag.color)
    tag.description = data.get('description', tag.description)
    tag.parent_id = data.get('parent_id', tag.parent_id)
    
    # Update synonyms
    new_synonyms = data.get('synonyms', [])
    existing_synonyms = [syn.synonym for syn in tag.synonyms]
    
    # Remove deleted synonyms
    for syn in tag.synonyms:
        if syn.synonym not in new_synonyms:
            db.session.delete(syn)
    
    # Add new synonyms
    for syn in new_synonyms:
        if syn not in existing_synonyms:
            db.session.add(TagSynonym(
                tag_id=tag_id,
                synonym=syn
            ))
    
    try:
        db.session.commit()
        return jsonify(tag.to_dict(True, True))
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags/<int:tag_id>', methods=['DELETE'])
def delete_tag(tag_id):
    """Delete tag."""
    tag = Tag.query.get_or_404(tag_id)
    
    try:
        # Remove tag from all documents first
        DocumentTag.query.filter_by(tag_id=tag_id).delete()
        db.session.delete(tag)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<int:doc_id>/tags', methods=['PUT'])
def update_document_tags(doc_id):
    """Update document tags."""
    doc = Document.query.get_or_404(doc_id)
    data = request.get_json()
    tag_ids = data.get('tag_ids', [])
    
    try:
        # Remove existing tags
        DocumentTag.query.filter_by(document_id=doc_id).delete()
        
        # Add new tags
        for tag_id in tag_ids:
            tag = Tag.query.get(tag_id)
            if tag:
                doc.tags.append(tag)
        
        db.session.commit()
        return jsonify(doc.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/llms')
def get_llms():
    """Get list of available LLM models."""
    return jsonify(list(get_all_llms().values()))

@app.route('/api/validation-rules')
def get_validation_rules():
    """Get query validation rules."""
    return jsonify({
        'min_length': QUERY_VALIDATION.min_length,
        'max_length': QUERY_VALIDATION.max_length,
        'required_keywords': QUERY_VALIDATION.required_keywords,
        'forbidden_keywords': QUERY_VALIDATION.forbidden_keywords
    })

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Handle document upload for query."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    permanent = request.form.get('permanent', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(filepath)
        
        # Process the document
        from .document_processor import process_document
        metadata = {
            "title": filename,
            "source": "user_upload",
            "classification": "public",
            "is_temporary": not permanent
        }
        
        process_document(
            filepath,
            output_dir="processed",
            metadata=metadata
        )
        
        # If permanent, add to database
        if permanent:
            session = Session()
            document = Document(
                path=filepath,
                title=filename,
                classification="public"
            )
            session.add(document)
            session.commit()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'permanent': permanent,
            'path': filepath
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_llm():
    """Process LLM query."""
    data = request.get_json()
    query = data.get('query')
    llm_model = data.get('llm_model')
    documents = data.get('documents', [])
    format_options = data.get('format', {})
    
    if not query or not llm_model:
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Get LLM configuration
        llm_config = get_llm_config(llm_model)
        
        # Here you would integrate with your actual LLM service
        # For now, we'll return a mock response
        response = {
            'response': f"Query received: {query}\n\nLLM model: {llm_config['name']}\n\nProcessing...",
            'model': llm_config['name'],
            'sources': ['Document 1', 'Document 2'] + [doc['filename'] for doc in documents],
            'confidence': 0.95
        }
        
        # Format response based on preferences
        if format_options.get('markdown', False):
            response['response'] = response['response'].replace('\n', '<br>')
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics and visualizations."""
    session = Session()
    
    # Get total documents and versions
    total_docs = session.query(func.count(Document.id)).scalar()
    total_versions = session.query(func.count(Version.id)).scalar()
    
    # Get version distribution
    version_dist = pd.read_sql(
        session.query(
            Version.document_id,
            func.count(Version.id).label('version_count')
        ).group_by(Version.document_id).subquery(),
        session.bind
    )
    
    # Create version distribution plot
    fig = go.Figure(
        data=[go.Histogram(x=version_dist['version_count'], nbinsx=10)],
        layout=go.Layout(
            title='Version Distribution',
            xaxis_title='Number of Versions per Document',
            yaxis_title='Document Count'
        )
    )
    version_dist_plot = fig.to_html(full_html=False)
    
    # Get classification distribution
    class_dist = pd.read_sql(
        session.query(
            Document.classification,
            func.count(Document.id).label('count')
        ).group_by(Document.classification).subquery(),
        session.bind
    )
    
    # Create classification pie chart
    fig = go.Figure(
        data=[go.Pie(labels=class_dist['classification'], values=class_dist['count'])],
        layout=go.Layout(
            title='Document Classification Distribution'
        )
    )
    class_dist_plot = fig.to_html(full_html=False)
    
    return jsonify({
        'total_docs': total_docs,
        'total_versions': total_versions,
        'version_dist_plot': version_dist_plot,
        'class_dist_plot': class_dist_plot
    })

@app.route('/api/versions/<int:doc_id>')
def get_versions(doc_id):
    """Get version history via API."""
    session = Session()
    versions = session.query(Version).filter_by(document_id=doc_id).order_by(Version.version_number.desc()).all()
    return jsonify([{
        'id': v.id,
        'version': v.version_number,
        'timestamp': v.timestamp.isoformat(),
        'file_size': v.file_size,
        'changes': json.loads(v.changes) if v.changes else None
    } for v in versions])

if __name__ == '__main__':
    app.run(debug=True)
