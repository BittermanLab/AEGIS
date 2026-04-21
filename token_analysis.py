#!/usr/bin/env python3
"""
Detailed token analysis script for agent I/O logs.
"""

import json

def estimate_tokens(text):
    """Rough token estimation: ~4 characters per token"""
    if not text:
        return 0
    return len(str(text)) // 4

def analyze_tokens():
    log_file_path = '/home/jackg/irae_graph/logs/agent_io/agent_io_20250603_162815.jsonl'
    
    # Create detailed breakdown table
    agent_data = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    
                    agent_name = entry.get('agent_name', 'unknown')
                    event_type = entry.get('event_type') or 'none'
                    system_prompt = entry.get('system_prompt', '')
                    user_prompt = entry.get('prompt', '')
                    token_usage = entry.get('token_usage', {})
                    
                    # Categorize agent type
                    if 'Note Extractor' in agent_name:
                        agent_type = 'Note Extractor'
                        stage = '1-Extraction'
                    elif 'Balanced Event Identifier' in agent_name:
                        agent_type = 'Balanced Identifier'
                        stage = '2-Identification'
                    elif 'Comprehensive Event Identifier' in agent_name:
                        agent_type = 'Comprehensive Identifier'
                        stage = '2-Identification'
                    elif 'Specific Event Identifier' in agent_name:
                        agent_type = 'Specific Identifier'
                        stage = '2-Identification'
                    elif 'Event Identification Judge' in agent_name:
                        agent_type = 'Identification Judge'
                        stage = '3-Judge'
                    elif 'Temporality Classifier' in agent_name:
                        agent_type = 'Temporality Classifier'
                        stage = '4-Temporality'
                    elif 'Conservative Grader' in agent_name:
                        agent_type = 'Conservative Grader'
                        stage = '5-Grading'
                    elif 'Evidence-based Grader' in agent_name:
                        agent_type = 'Evidence-based Grader'
                        stage = '5-Grading'
                    elif 'Guidelines-focused Grader' in agent_name:
                        agent_type = 'Guidelines-focused Grader'
                        stage = '5-Grading'
                    elif 'Grading Judge' in agent_name:
                        agent_type = 'Grading Judge'
                        stage = '6-Grading Judge'
                    elif 'Temporality Judge' in agent_name:
                        agent_type = 'Temporality Judge'
                        stage = '7-Temporal Judge'
                    else:
                        agent_type = 'Other'
                        stage = '8-Other'
                    
                    agent_data.append({
                        'stage': stage,
                        'agent_type': agent_type,
                        'event_type': event_type,
                        'agent_name': agent_name,
                        'actual_prompt_tokens': token_usage.get('prompt_tokens', 0),
                        'actual_completion_tokens': token_usage.get('completion_tokens', 0),
                        'estimated_system_tokens': estimate_tokens(system_prompt),
                        'estimated_user_tokens': estimate_tokens(user_prompt),
                        'system_prompt_chars': len(system_prompt),
                        'user_prompt_chars': len(user_prompt),
                        'total_cost': token_usage.get('total_cost', 0.0)
                    })

        # Sort by stage, event type, then agent type
        agent_data.sort(key=lambda x: (x['stage'], x['event_type'], x['agent_type']))

        print('='*160)
        print('DETAILED TOKEN BREAKDOWN BY AGENT, CONDITION, AND STAGE')
        print('='*160)
        print(f"{'Stage':<20} {'Agent Type':<25} {'Event':<12} {'Prompt':<7} {'Output':<7} {'Est.Sys':<7} {'Est.User':<8} {'SysChars':<8} {'Cost':<8}")
        print('-'*160)
        
        current_stage = None
        stage_totals = {}
        
        for data in agent_data:
            stage = data['stage']
            
            # Track stage totals
            if stage not in stage_totals:
                stage_totals[stage] = {
                    'prompt': 0, 'completion': 0, 'est_sys': 0, 'est_user': 0, 'cost': 0.0, 'count': 0
                }
            
            stage_totals[stage]['prompt'] += data['actual_prompt_tokens']
            stage_totals[stage]['completion'] += data['actual_completion_tokens']
            stage_totals[stage]['est_sys'] += data['estimated_system_tokens']
            stage_totals[stage]['est_user'] += data['estimated_user_tokens']
            stage_totals[stage]['cost'] += data['total_cost']
            stage_totals[stage]['count'] += 1
            
            # Print stage separator
            if current_stage != stage:
                if current_stage is not None:
                    print('-'*160)
                current_stage = stage
            
            print(f"{stage:<20} {data['agent_type']:<25} {data['event_type']:<12} {data['actual_prompt_tokens']:<7} {data['actual_completion_tokens']:<7} {data['estimated_system_tokens']:<7} {data['estimated_user_tokens']:<8} {data['system_prompt_chars']:<8} ${data['total_cost']:<7.4f}")
        
        print('='*160)
        print('STAGE SUMMARIES:')
        print('='*160)
        print(f"{'Stage':<20} {'Calls':<6} {'Prompt':<8} {'Output':<8} {'Est.Sys':<8} {'Est.User':<9} {'Cost':<8} {'%Total':<6}")
        print('-'*160)
        
        total_prompt = sum(st['prompt'] for st in stage_totals.values())
        total_completion = sum(st['completion'] for st in stage_totals.values())
        total_cost = sum(st['cost'] for st in stage_totals.values())
        
        for stage in sorted(stage_totals.keys()):
            st = stage_totals[stage]
            pct = (st['prompt'] + st['completion']) / (total_prompt + total_completion) * 100
            print(f"{stage:<20} {st['count']:<6} {st['prompt']:<8} {st['completion']:<8} {st['est_sys']:<8} {st['est_user']:<9} ${st['cost']:<7.4f} {pct:<5.1f}%")
        
        print('-'*160)
        print(f"{'TOTAL':<20} {sum(st['count'] for st in stage_totals.values()):<6} {total_prompt:<8} {total_completion:<8} {sum(st['est_sys'] for st in stage_totals.values()):<8} {sum(st['est_user'] for st in stage_totals.values()):<9} ${total_cost:<7.4f} 100.0%")
        
        # Additional breakdown by event type
        print('\n' + '='*160)
        print('BREAKDOWN BY EVENT TYPE:')
        print('='*160)
        
        event_totals = {}
        for data in agent_data:
            event = data['event_type']
            if event not in event_totals:
                event_totals[event] = {'prompt': 0, 'completion': 0, 'count': 0, 'cost': 0.0}
            event_totals[event]['prompt'] += data['actual_prompt_tokens']
            event_totals[event]['completion'] += data['actual_completion_tokens']
            event_totals[event]['count'] += 1
            event_totals[event]['cost'] += data['total_cost']
        
        print(f"{'Event Type':<15} {'Calls':<6} {'Prompt':<8} {'Output':<8} {'Total':<8} {'Cost':<8} {'%Total':<6}")
        print('-'*80)
        
        for event in sorted(event_totals.keys()):
            et = event_totals[event]
            total_tokens = et['prompt'] + et['completion']
            pct = total_tokens / (total_prompt + total_completion) * 100
            print(f"{event:<15} {et['count']:<6} {et['prompt']:<8} {et['completion']:<8} {total_tokens:<8} ${et['cost']:<7.4f} {pct:<5.1f}%")

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    analyze_tokens()