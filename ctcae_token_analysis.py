#!/usr/bin/env python3
"""
CTCAE-focused token analysis script.
"""

import json

def analyze_ctcae_tokens():
    log_file_path = '/home/jackg/irae_graph/logs/agent_io/agent_io_20250603_165915.jsonl'
    
    # Focus on where CTCAE definitions appear
    ctcae_analysis = {}
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    
                    agent_name = entry.get('agent_name', 'unknown')
                    event_type = entry.get('event_type') or 'none'
                    system_prompt = entry.get('system_prompt', '')
                    token_usage = entry.get('token_usage', {})
                    
                    # Categorize by what CTCAE content they get
                    if 'Note Extractor' in agent_name:
                        ctcae_content = 'None'
                        agent_category = 'Note Extractor'
                    elif any(x in agent_name for x in ['Identifier', 'Judge']):
                        if 'Grader' in agent_name or 'Grading' in agent_name:
                            ctcae_content = 'Full CTCAE Grading Criteria'
                            agent_category = 'Grader/Judge (Full CTCAE)'
                        else:
                            ctcae_content = 'CTCAE Term Names Only'
                            agent_category = 'Identifier/Judge (Terms Only)'
                    elif 'Grader' in agent_name:
                        ctcae_content = 'Full CTCAE Grading Criteria'
                        agent_category = 'Grader (Full CTCAE)'
                    elif 'Temporality' in agent_name:
                        ctcae_content = 'CTCAE Term Names Only'
                        agent_category = 'Temporality (Terms Only)'
                    else:
                        ctcae_content = 'Unknown'
                        agent_category = 'Other'
                    
                    key = (agent_category, event_type)
                    if key not in ctcae_analysis:
                        ctcae_analysis[key] = {
                            'count': 0,
                            'total_prompt_tokens': 0,
                            'total_system_chars': 0,
                            'total_cost': 0.0,
                            'ctcae_content': ctcae_content,
                            'example_agent': agent_name
                        }
                    
                    ctcae_analysis[key]['count'] += 1
                    ctcae_analysis[key]['total_prompt_tokens'] += token_usage.get('prompt_tokens', 0)
                    ctcae_analysis[key]['total_system_chars'] += len(system_prompt)
                    ctcae_analysis[key]['total_cost'] += token_usage.get('total_cost', 0.0)

        print('='*140)
        print('CTCAE TOKEN USAGE BREAKDOWN')
        print('='*140)
        print(f"{'Agent Category':<35} {'Event':<12} {'CTCAE Content':<30} {'Calls':<6} {'Prompt':<8} {'SysChars':<8} {'Cost':<8}")
        print('-'*140)
        
        # Sort by agent category and event
        for key in sorted(ctcae_analysis.keys()):
            data = ctcae_analysis[key]
            agent_category, event_type = key
            
            print(f"{agent_category:<35} {event_type:<12} {data['ctcae_content']:<30} {data['count']:<6} {data['total_prompt_tokens']:<8} {data['total_system_chars']:<8} ${data['total_cost']:<7.4f}")
        
        print('='*140)
        
        # Summary by CTCAE content type
        print('\nSUMMARY BY CTCAE CONTENT TYPE:')
        print('='*80)
        
        content_summary = {}
        for key, data in ctcae_analysis.items():
            content_type = data['ctcae_content']
            if content_type not in content_summary:
                content_summary[content_type] = {
                    'count': 0, 'prompt_tokens': 0, 'system_chars': 0, 'cost': 0.0
                }
            
            content_summary[content_type]['count'] += data['count']
            content_summary[content_type]['prompt_tokens'] += data['total_prompt_tokens']
            content_summary[content_type]['system_chars'] += data['total_system_chars']
            content_summary[content_type]['cost'] += data['total_cost']
        
        print(f"{'CTCAE Content Type':<35} {'Calls':<6} {'Prompt':<8} {'SysChars':<10} {'Cost':<8}")
        print('-'*80)
        
        for content_type in sorted(content_summary.keys()):
            data = content_summary[content_type]
            print(f"{content_type:<35} {data['count']:<6} {data['prompt_tokens']:<8} {data['system_chars']:<10} ${data['cost']:<7.4f}")
        
        # Calculate average system prompt sizes
        print('\nAVERAGE SYSTEM PROMPT SIZE BY CTCAE CONTENT:')
        print('='*70)
        print(f"{'CTCAE Content Type':<35} {'Avg Chars':<10} {'Avg Tokens':<10}")
        print('-'*70)
        
        for content_type, data in content_summary.items():
            if data['count'] > 0:
                avg_chars = data['system_chars'] // data['count']
                avg_tokens = avg_chars // 4  # Rough estimation
                print(f"{content_type:<35} {avg_chars:<10} {avg_tokens:<10}")

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    analyze_ctcae_tokens()