#!/usr/bin/env python3
"""
Simple test script for CSV data mapping detection
"""

from src.utils.file_parsers import FileParser

def test_csv_mapping():
    """Test the enhanced CSV parsing with mapping detection"""
    
    parser = FileParser()
    csv_file = 'sample_data_mapping.csv'
    
    try:
        print('🔍 Testing Enhanced CSV Mapping Detection')
        print('=' * 50)
        
        result = parser.parse_csv(csv_file)
        
        print(f'📄 File: {csv_file}')
        print(f'📊 Result type: {type(result)}')
        print(f'📋 Row count: {len(result)}')
        
        if result and result[0].get('_mapping_type') == 'data_mapping':
            print('\n✅ DETECTED AS DATA MAPPING CSV')
            
            print('\n📈 Sample Mapping Relationships:')
            print('-' * 60)
            
            for i, row in enumerate(result[:3], 1):
                mapping_rel = row.get('_mapping_relationship', {})
                if mapping_rel:
                    print(f'{i}. Source: {mapping_rel.get("source", "N/A")}')
                    print(f'   Target: {mapping_rel.get("target", "N/A")}')
                    print(f'   Transformation: {mapping_rel.get("transformation", "N/A")}')
                    print(f'   Type: {mapping_rel.get("data_type", "N/A")}')
                    print()
            
            print(f'📊 Total mappings detected: {len(result)}')
        else:
            print('\n❌ NOT DETECTED AS MAPPING CSV')
    
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_csv_mapping()
