from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.String(255), nullable=True)
    color = db.Column(db.String(7), nullable=True)  # Hex color code
    parent_id = db.Column(db.Integer, db.ForeignKey('tag.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    usage_count = db.Column(db.Integer, nullable=False, default=0)
    
    # Relationships
    parent = db.relationship('Tag', remote_side=[id], backref=db.backref('children', lazy=True))
    synonyms = db.relationship('TagSynonym', back_populates='tag')
    
    def __repr__(self):
        return f'<Tag {self.name}>'

    def to_dict(self, include_children=False, include_synonyms=False):
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'color': self.color,
            'created_at': self.created_at.isoformat(),
            'usage_count': self.usage_count,
            'parent_id': self.parent_id,
            'parent': self.parent.name if self.parent else None
        }
        
        if include_children:
            data['children'] = [child.to_dict() for child in self.children]
            
        if include_synonyms:
            data['synonyms'] = [synonym.to_dict() for synonym in self.synonyms]
            
        return data

    def validate(self, all_tags=None):
        """Validate this tag instance."""
        from ..tools.tag_validator import TagValidator
        validator = TagValidator()
        
        tag_data = {
            'name': self.name,
            'color': self.color,
            'parent_id': self.parent_id
        }
        
        return validator.validate_tag(tag_data, all_tags or Tag.query.all())

class TagSynonym(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'), nullable=False)
    synonym = db.Column(db.String(50), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    tag = db.relationship('Tag', back_populates='synonyms')
    
    def __repr__(self):
        return f'<TagSynonym {self.synonym}>'

    def to_dict(self):
        return {
            'id': self.id,
            'synonym': self.synonym,
            'created_at': self.created_at.isoformat()
        }

class DocumentTag(db.Model):
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'), primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    document = db.relationship('Document', backref=db.backref('document_tags', lazy=True))
    tag = db.relationship('Tag', backref=db.backref('document_tags', lazy=True))
    
    def __repr__(self):
        return f'<DocumentTag {self.document_id} - {self.tag_id}>'

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255), nullable=False)
    classification = db.Column(db.String(50), nullable=False, default='public')
    description = db.Column(db.Text, nullable=True)
    keywords = db.Column(db.String(255), nullable=True)
    version = db.Column(db.Integer, nullable=False, default=1)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    size = db.Column(db.Integer, nullable=False)
    
    # Relationships
    tags = db.relationship('Tag', secondary='document_tag', backref=db.backref('documents', lazy=True))
    
    def __repr__(self):
        return f'<Document {self.filename} v{self.version}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'version': self.version,
            'classification': self.classification,
            'description': self.description,
            'keywords': self.keywords,
            'uploaded_at': self.uploaded_at.isoformat(),
            'size': self.size,
            'tags': [tag.to_dict() for tag in self.tags]
        }
