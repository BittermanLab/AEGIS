# Debug Tools for OpenAI Agent SDK

This directory contains tools specifically for debugging the OpenAI Agent SDK workflow.

## Debugging a Single Note

To debug a single clinical note and see detailed agent inputs/outputs, use the `debug_note.py` script:

```bash
# Run with a file
python -m graphs.openai_agent.debug_note --note-file path/to/note.txt --detailed

# Run with stdin (paste a note)
python -m graphs.openai_agent.debug_note --detailed
```

### Command Line Options

- `--note-file PATH`: Path to a file containing the clinical note
- `--model-config NAME`: Model configuration to use (default, o3_mini, etc.)
- `--prompt-variant NAME`: Prompt variant to use (default, detailed, etc.)
- `--output-file PATH`: Path for the JSON output file (default: debug_output.json)
- `--patient-id ID`: Patient ID for the note (default: DEBUG)
- `--timepoint TIME`: Timepoint for the note (default: t1)
- `--detailed`: Enable highly detailed debugging output
- `--event-types LIST`: Comma-separated list of event types to process

### Debugging Output

The script outputs:

1. `debug_output.json`: Contains the final prediction result
2. Detailed logging information to the console (stdout) when running with --detailed

### Configuring Log Levels

For more control over logging levels:

```bash
# Set token tracking verbosity via environment variable
TOKEN_TRACKER_LOG_LEVEL=DEBUG python -m graphs.openai_agent.debug_note --note-file note.txt

# Or when running a sweep
python scripts/run_sweep.py --sweep dev_agent --log_level DEBUG
```

You can also set Python logging programmatically:

```python
import logging
logging.getLogger('graphs.openai_agent').setLevel(logging.DEBUG)
logging.getLogger('graphs.openai_agent.parallel_agents').setLevel(logging.DEBUG)
```

### Identifying Problems

To identify why certain events are being predicted incorrectly:

1. Run the debug script with a note where you know the ground truth
2. Look at the debug logs in the console to see what each agent is doing
3. Focus on:
   - Event identification: Is the event correctly identified in the first place?
   - Grading: Is the severity grade correct?
   - Attribution: Is the causality attribution correct?
   - Certainty: Is the certainty level correct?

Common issues include:
- Misidentified events due to ambiguous language
- Incorrect grading due to missing or misinterpreted symptoms
- Incorrect attribution of events to non-treatment causes
- Wrong temporal classification (past vs. current events)

## Configuration

The model configurations are defined in:
- `parallel_agents/utils/model_config.py`

The current configurations use:
- `gpt-4o-mini-jg`: Default model with temperature=0
- `o3-mini-jg`: Alternative model (cannot use temperature)

## Azure Configuration

The system uses Azure OpenAI with the following JG deployment names:
- gpt-4o-mini-jg
- o3-mini-jg

These are mapped in the model_config.py file.

## Token Usage

To see detailed token usage information for a run:

1. Run the debug script with `--detailed` or set `TOKEN_TRACKER_LOG_LEVEL=DEBUG`
2. Check the console output for token usage information
3. The debug_output.json file will also contain a token_usage section