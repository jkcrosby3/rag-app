from typing import Dict, List, Optional
from datetime import datetime
import re
from .metadata_validator import MetadataSchema

class TagValidator:
    """Validator for tag-related operations."""
    
    # Tag name validation rules
    TAG_NAME_PATTERN = r'^[a-zA-Z0-9_\-]+$'  # Alphanumeric with underscores and hyphens
    MAX_TAG_NAME_LENGTH = 50
    MIN_TAG_NAME_LENGTH = 2
    
    # Color validation rules
    COLOR_PATTERN = r'^#[0-9a-fA-F]{6}$'  # Hex color code
    
    # Synonym validation rules
    MAX_SYNONYMS_PER_TAG = 5
    MAX_SYNONYM_LENGTH = 50
    
    # Hierarchy validation rules
    MAX_HIERARCHY_DEPTH = 5
    
    # Performance validation rules
    MAX_DOCUMENTS_PER_TAG = 10000
    MIN_DOCUMENTS_PER_TAG = 1
    MAX_SYNONYMS_FOR_PERFORMANCE = 10
    
    # Data quality validation rules
    MIN_TAG_DESCRIPTION_LENGTH = 10
    REQUIRED_TAG_PREFIXES = ['project-', 'department-', 'category-']
    
    # Security validation rules
    MAX_TAG_NAME_LENGTH_FOR_SECURITY = 255
    SQL_INJECTION_PATTERNS = [
        r'--',  # SQL comment
        r'/*',  # SQL comment start
        r'*/',  # SQL comment end
        r';',   # SQL statement separator
        r'\',  # SQL escape character
        r'\x', # Hex escape
        r'\u', # Unicode escape
    ]
    
    # Business rules validation
    MANDATORY_TAGS_FOR_DOCUMENT_TYPES = {
        'project': ['project-', 'department-'],
        'department': ['department-', 'category-'],
        'category': ['category-', 'project-']
    }
    
    def validate_tag_name(self, tag_name: str, document_type: Optional[str] = None) -> Dict[str, str]:
        """Validate tag name format."""
        errors = {}
        
        if not tag_name:
            errors['name'] = 'Tag name is required. Please enter a name for the tag.'
            return errors
            
        # Security validation
        if len(tag_name) > self.MAX_TAG_NAME_LENGTH_FOR_SECURITY:
            errors['name'] = f'Security warning: Tag name is too long ({len(tag_name)} characters). ' \
                         'Long tag names may impact database performance.'
            
        # Check for SQL injection patterns
        if any(pattern in tag_name for pattern in self.SQL_INJECTION_PATTERNS):
            errors['name'] = 'Security warning: Tag name contains potentially unsafe characters. ' \
                         'Please avoid special characters that could be used for SQL injection.'
            
        if len(tag_name) < self.MIN_TAG_NAME_LENGTH:
            errors['name'] = f'Tag name must be at least {self.MIN_TAG_NAME_LENGTH} characters long. ' \
                         f'For example: "{tag_name}" is too short.'
        
        if len(tag_name) > self.MAX_TAG_NAME_LENGTH:
            errors['name'] = f'Tag name cannot exceed {self.MAX_TAG_NAME_LENGTH} characters. ' \
                         f'Current length: {len(tag_name)} characters.'
        
        if not re.match(self.TAG_NAME_PATTERN, tag_name):
            errors['name'] = 'Tag name must contain only letters, numbers, underscores, and hyphens. ' \
                         'Invalid characters found: ' + ''.join(
                             char for char in tag_name 
                             if not re.match(r'[a-zA-Z0-9_\-]', char)
                         )
        
        return errors
    
    def validate_tag_color(self, color: str) -> Dict[str, str]:
        """Validate tag color format."""
        errors = {}
        
        if color and not re.match(self.COLOR_PATTERN, color):
            errors['color'] = 'Color must be a valid hex code (e.g., #FFFFFF). ' \
                         'Current value: {color}. ' \
                         'Please use a 6-digit hex code starting with #.'
        
        return errors
    
    def validate_synonym(self, synonym: str, tag_name: str) -> Dict[str, str]:
        """Validate a single synonym."""
        errors = {}
        
        if not synonym:
            errors['synonym'] = 'Synonym cannot be empty. Please enter a synonym for the tag.'
            return errors
            
        if len(synonym) > self.MAX_SYNONYM_LENGTH:
            errors['synonym'] = f'Synonym cannot exceed {self.MAX_SYNONYM_LENGTH} characters. ' \
                             f'Current length: {len(synonym)} characters.'
        
        # Synonyms should be distinct from the tag name
        if synonym.lower() == tag_name.lower():
            errors['synonym'] = f'Synonym cannot be the same as the tag name "{tag_name}". ' \
                             'Please choose a different synonym.'
        
        return errors
    
    def validate_synonyms(self, synonyms: List[str]) -> Dict[str, str]:
        """Validate list of synonyms."""
        errors = {}
        
        if len(synonyms) > self.MAX_SYNONYMS_PER_TAG:
            errors['synonyms'] = f'A tag cannot have more than {self.MAX_SYNONYMS_PER_TAG} synonyms'
        
        # Check for duplicates
        if len(set(synonyms)) != len(synonyms):
            duplicates = [s for s in synonyms if synonyms.count(s) > 1]
            errors['synonyms'] = f'Duplicate synonyms found: {", ".join(duplicates)}. ' \
                              'Each synonym must be unique for a tag.'
            
        # Check for duplicate synonyms across different tags
        if len(synonyms) > 1:
            other_tags = Tag.query.filter(Tag.id != tag_id).all()
            for syn in synonyms:
                if any(syn.lower() == s.synonym.lower() 
                       for tag in other_tags 
                       for s in tag.synonyms):
                    errors['synonyms'] = f'Synonym "{syn}" already exists for another tag. ' \
                                      'Please choose a unique synonym.'
                    break
            
        # Performance validation
        if len(synonyms) > self.MAX_SYNONYMS_FOR_PERFORMANCE:
            errors['synonyms'] = f'Too many synonyms ({len(synonyms)}). ' \
                              'This may impact search performance. ' \
                              'Maximum recommended: {self.MAX_SYNONYMS_FOR_PERFORMANCE}.'
        
        # Validate each synonym
        for synonym in synonyms:
            synonym_errors = self.validate_synonym(synonym)
            if synonym_errors:
                errors.update(synonym_errors)
        
        return errors
    
    def validate_hierarchy(self, tag: Dict[str, any], all_tags: List[Dict[str, any]]) -> Dict[str, str]:
        """Validate tag hierarchy."""
        errors = {}
        
        if tag.get('parent_id'):
            # Check if parent exists
            parent = next((t for t in all_tags if t['id'] == tag['parent_id']), None)
            if not parent:
                errors['parent_id'] = f'Parent tag with ID {tag["parent_id"]} does not exist. ' \
                                   'Please select a valid parent tag from the dropdown.'
                return errors
                
            # Check hierarchy depth
            depth = 1
            current = parent
            while current.get('parent_id'):
                current = next((t for t in all_tags if t['id'] == current['parent_id']), None)
                if not current:
                    break
                depth += 1
                
            if depth >= self.MAX_HIERARCHY_DEPTH:
                errors['parent_id'] = f'The maximum hierarchy depth of {self.MAX_HIERARCHY_DEPTH} levels has been exceeded. ' \
                                   f'Current depth: {depth} levels. ' \
                                   'Please choose a parent tag higher in the hierarchy.'
        
        return errors
    
    def validate_tag(self, tag_data: Dict[str, any], all_tags: Optional[List[Dict[str, any]]] = None, 
                    document_count: Optional[int] = None, document_type: Optional[str] = None) -> Dict[str, str]:
        """Validate a complete tag object."""
        errors = {}
        
        # Validate required fields
        errors.update(self.validate_tag_name(tag_data.get('name', ''), document_type))
        
        # Validate optional fields
        errors.update(self.validate_tag_color(tag_data.get('color', '')))
        
        # Validate description
        description = tag_data.get('description', '')
        if description and len(description) < self.MIN_TAG_DESCRIPTION_LENGTH:
            errors['description'] = f'Description must be at least {self.MIN_TAG_DESCRIPTION_LENGTH} characters long. ' \
                                 f'Current length: {len(description)} characters.'
        
        # Validate synonyms if present
        synonyms = tag_data.get('synonyms', [])
        if synonyms:
            errors.update(self.validate_synonyms(synonyms, tag_data.get('name', ''), tag_data.get('id', 0)))
        
        # Validate hierarchy if parent is specified
        if all_tags and tag_data.get('parent_id'):
            errors.update(self.validate_hierarchy(tag_data, all_tags))
            
        # Validate document count
        if document_count is not None:
            if document_count > self.MAX_DOCUMENTS_PER_TAG:
                errors['documents'] = f'This tag has too many documents ({document_count}). ' \
                                   f'Maximum recommended: {self.MAX_DOCUMENTS_PER_TAG}. ' \
                                   'Consider creating sub-tags or splitting the tag.'
            elif document_count < self.MIN_DOCUMENTS_PER_TAG:
                errors['documents'] = f'This tag has very few documents ({document_count}). ' \
                                   'Consider merging with other tags or removing if unused.'
            
        # Validate business rules
        if document_type and document_type in self.MANDATORY_TAGS_FOR_DOCUMENT_TYPES:
            required_prefixes = self.MANDATORY_TAGS_FOR_DOCUMENT_TYPES[document_type]
            if not any(tag_data['name'].startswith(prefix) for prefix in required_prefixes):
                errors['name'] = f'For {document_type} documents, tag name must start with one of: ' \
                               f'{", ".join(required_prefixes)}. ' \
                               f'Current name: "{tag_data["name"]}".'
        
        # Validate hierarchy if parent is specified
        if all_tags and tag_data.get('parent_id'):
            errors.update(self.validate_hierarchy(tag_data, all_tags))
        
        return errors
    
    def validate_tag_update(self, tag_data: Dict[str, any], existing_tag: Dict[str, any], all_tags: List[Dict[str, any]]) -> Dict[str, str]:
        """Validate tag update operation."""
        errors = {}
        
        # Validate name change
        if 'name' in tag_data:
            errors.update(self.validate_tag_name(tag_data['name']))
            
            # Check if name is already in use
            if any(t['name'].lower() == tag_data['name'].lower() and t['id'] != existing_tag['id'] for t in all_tags):
                existing = next(t for t in all_tags if t['name'].lower() == tag_data['name'].lower())
                errors['name'] = f'Tag name "{tag_data["name"]}" is already in use by tag ID {existing["id"]}. ' \
                               'Please choose a unique name for your tag.'
        
        # Validate parent change
        if 'parent_id' in tag_data:
            errors.update(self.validate_hierarchy({'parent_id': tag_data['parent_id']}, all_tags))
            
            # Check for circular references
            if tag_data['parent_id'] == existing_tag['id']:
                errors['parent_id'] = 'A tag cannot be its own parent. Please choose a different parent tag.'
                
            # Check if parent is a child of this tag
            current = next((t for t in all_tags if t['id'] == tag_data['parent_id']), None)
            while current:
                if current['id'] == existing_tag['id']:
                    errors['parent_id'] = 'Cannot create circular reference in hierarchy. ' \
                                       f'Tag {existing_tag["name"]} is already a parent of ' \
                                       f'the selected tag {current["name"]}. '
                    break
                current = next((t for t in all_tags if t['id'] == current.get('parent_id')), None)
        
        return errors
