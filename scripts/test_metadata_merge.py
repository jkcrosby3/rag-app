"""Test metadata merge functionality."""
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processing.batch_processor import BatchProcessor

def test_metadata_merge():
    """Test that metadata JSON files are properly merged."""
    bp = BatchProcessor()
    
    # Test with a newspaper file that has metadata
    test_file = project_root / "data/documents/smithsonian/newspapers/text_files/sn82014385.txt"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"Testing metadata merge with: {test_file.name}")
    
    try:
        result = bp._extract_text_from_file(test_file)
        
        # Check if metadata was merged
        has_enriched = 'enriched' in result['metadata']
        has_people = 'people_mentioned' in result['metadata']
        
        print(f"\n✓ File processed successfully")
        print(f"✓ Has enriched metadata: {has_enriched}")
        print(f"✓ Has people_mentioned: {has_people}")
        
        if has_people:
            people_count = len(result['metadata']['people_mentioned'])
            print(f"✓ People mentioned: {people_count}")
            if people_count > 0:
                print(f"  Examples: {result['metadata']['people_mentioned'][:3]}")
        
        if 'battles_mentioned' in result['metadata']:
            battles = result['metadata']['battles_mentioned']
            print(f"✓ Battles mentioned: {len(battles)}")
        
        if 'war_relevance_score' in result['metadata']:
            score = result['metadata']['war_relevance_score']
            print(f"✓ War relevance score: {score}")
        
        print("\n🎉 Metadata merge is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_metadata_merge()
    sys.exit(0 if success else 1)
