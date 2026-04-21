# IRAE Graph Architecture Variations

This document contains mermaid diagrams explaining the different architecture variations used in the IRAE Graph system for identifying and grading immune-related adverse events (IRAEs).

## 1. Standard Multi-Agent Architecture (Default)

The standard architecture uses multiple agents with judge-based consensus for both identification and grading phases.

```mermaid
graph TD
    A[Clinical Note] --> B[Event Processing Pipeline]
    
    subgraph "1. Identification Phase"
        B --> C1[Identifier Agent 1]
        B --> C2[Identifier Agent 2] 
        B --> C3[Identifier Agent 3]
        C1 --> D[Identification Judge]
        C2 --> D
        C3 --> D
        D --> E[Aggregated Identification]
    end
    
    subgraph "2. Grading Phase"
        E --> F1[Past Events?]
        E --> G1[Current Events?]
        
        F1 -->|Yes| F2[Past Grader 1]
        F1 -->|Yes| F3[Past Grader 2]
        F1 -->|Yes| F4[Past Grader 3]
        F2 --> F5[Past Grading Judge]
        F3 --> F5
        F4 --> F5
        F5 --> H[Past Grade]
        
        G1 -->|Yes| G2[Current Grader 1]
        G1 -->|Yes| G3[Current Grader 2]
        G1 -->|Yes| G4[Current Grader 3]
        G2 --> G5[Current Grading Judge]
        G3 --> G5
        G4 --> G5
        G5 --> I[Current Grade]
    end
    
    subgraph "3. Attribution & Certainty"
        H --> J[Attribution Agent]
        I --> J
        H --> K[Certainty Agent]
        I --> K
        J --> L[Attribution Score]
        K --> M[Certainty Score]
    end
    
    subgraph "4. Meta-Evaluation"
        L --> N[Meta-Judge Agent]
        M --> N
        H --> N
        I --> N
        N --> O[Quality Assessment]
        O -->|Needs Improvement| B
        O -->|Satisfactory| P[Final Result]
    end
    
    P --> Q[Enhanced Event Result]
    
    style C1 fill:#e1f5fe
    style C2 fill:#e1f5fe
    style C3 fill:#e1f5fe
    style D fill:#fff3e0
    style F2 fill:#e8f5e8
    style F3 fill:#e8f5e8
    style F4 fill:#e8f5e8
    style F5 fill:#fff3e0
    style G2 fill:#e8f5e8
    style G3 fill:#e8f5e8
    style G4 fill:#e8f5e8
    style G5 fill:#fff3e0
    style N fill:#fce4ec
```

## 2. Single Agent Architecture (ablation_single)

This variation uses only one agent for identification and grading, removing the self-consistency mechanism.

```mermaid
graph TD
    A[Clinical Note] --> B[Event Processing Pipeline]
    
    subgraph "1. Identification Phase"
        B --> C1[Single Identifier Agent]
        C1 --> D[Identification Judge]
        D --> E[Aggregated Identification]
    end
    
    subgraph "2. Grading Phase"
        E --> F1[Past Events?]
        E --> G1[Current Events?]
        
        F1 -->|Yes| F2[Single Past Grader]
        F2 --> F5[Past Grading Judge]
        F5 --> H[Past Grade]
        
        G1 -->|Yes| G2[Single Current Grader]
        G2 --> G5[Current Grading Judge]
        G5 --> I[Current Grade]
    end
    
    subgraph "3. Attribution & Certainty"
        H --> J[Attribution Agent]
        I --> J
        H --> K[Certainty Agent]
        I --> K
        J --> L[Attribution Score]
        K --> M[Certainty Score]
    end
    
    subgraph "4. Meta-Evaluation"
        L --> N[Meta-Judge Agent]
        M --> N
        H --> N
        I --> N
        N --> O[Quality Assessment]
        O -->|Needs Improvement| B
        O -->|Satisfactory| P[Final Result]
    end
    
    P --> Q[Enhanced Event Result]
    
    style C1 fill:#ffebee
    style D fill:#fff3e0
    style F2 fill:#ffebee
    style F5 fill:#fff3e0
    style G2 fill:#ffebee
    style G5 fill:#fff3e0
    style N fill:#fce4ec
```

## 3. No-Judge Architecture (ablation_no_judge)

This variation removes judge agents and uses the first agent's result directly, eliminating consensus mechanisms.

```mermaid
graph TD
    A[Clinical Note] --> B[Event Processing Pipeline]
    
    subgraph "1. Identification Phase"
        B --> C1[Identifier Agent 1]
        B --> C2[Identifier Agent 2]
        B --> C3[Identifier Agent 3]
        C1 --> E[First Result Selected]
        C2 -.-> X1[Results Ignored]
        C3 -.-> X2[Results Ignored]
    end
    
    subgraph "2. Grading Phase"
        E --> F1[Past Events?]
        E --> G1[Current Events?]
        
        F1 -->|Yes| F2[Past Grader 1]
        F1 -->|Yes| F3[Past Grader 2]
        F1 -->|Yes| F4[Past Grader 3]
        F2 --> H[First Past Grade]
        F3 -.-> X3[Results Ignored]
        F4 -.-> X4[Results Ignored]
        
        G1 -->|Yes| G2[Current Grader 1]
        G1 -->|Yes| G3[Current Grader 2]
        G1 -->|Yes| G4[Current Grader 3]
        G2 --> I[First Current Grade]
        G3 -.-> X5[Results Ignored]
        G4 -.-> X6[Results Ignored]
    end
    
    subgraph "3. Attribution & Certainty"
        H --> J[Attribution Agent]
        I --> J
        H --> K[Certainty Agent]
        I --> K
        J --> L[Attribution Score]
        K --> M[Certainty Score]
    end
    
    subgraph "4. Meta-Evaluation"
        L --> N[Meta-Judge Agent]
        M --> N
        H --> N
        I --> N
        N --> O[Quality Assessment]
        O -->|Needs Improvement| B
        O -->|Satisfactory| P[Final Result]
    end
    
    P --> Q[Enhanced Event Result]
    
    style C1 fill:#e1f5fe
    style C2 fill:#ffebee
    style C3 fill:#ffebee
    style E fill:#ffcdd2
    style F2 fill:#e8f5e8
    style F3 fill:#ffebee
    style F4 fill:#ffebee
    style G2 fill:#e8f5e8
    style G3 fill:#ffebee
    style G4 fill:#ffebee
    style X1 fill:#f5f5f5
    style X2 fill:#f5f5f5
    style X3 fill:#f5f5f5
    style X4 fill:#f5f5f5
    style X5 fill:#f5f5f5
    style X6 fill:#f5f5f5
    style N fill:#fce4ec
```

## 4. Model Configuration Variations

Different models can be assigned to different agent types, allowing for specialized model selection.

```mermaid
graph TD
    A[Model Configuration] --> B[Agent Type Assignment]
    
    subgraph "Model Types"
        C1[GPT-4o]
        C2[GPT-4o-mini]
        C3[o1-preview]
        C4[o1-mini]
        C5[o3-mini]
    end
    
    subgraph "Agent Assignments"
        B --> D1[Identifier Agents]
        B --> D2[Judge Agents]
        B --> D3[Grader Agents]
        B --> D4[Attribution Agent]
        B --> D5[Certainty Agent]
        B --> D6[Meta-Judge Agent]
    end
    
    subgraph "Example: High-Performance Config"
        C3 --> D1
        C1 --> D2
        C1 --> D3
        C1 --> D4
        C1 --> D5
        C3 --> D6
    end
    
    subgraph "Example: Cost-Optimized Config"
        C2 --> D1
        C2 --> D2
        C2 --> D3
        C2 --> D4
        C2 --> D5
        C2 --> D6
    end
    
    style C1 fill:#e3f2fd
    style C2 fill:#e8f5e8
    style C3 fill:#fff3e0
    style C4 fill:#fce4ec
    style C5 fill:#f3e5f5
```

## 5. Temporal Processing Architecture

The system processes past and current events separately, allowing for temporal-specific analysis.

```mermaid
graph TD
    A[Identified Events] --> B{Event Temporal Classification}
    
    B --> C[Past Events Branch]
    B --> D[Current Events Branch]
    
    subgraph "Past Events Processing"
        C --> E1[Past Event Context]
        E1 --> F1[Past Grader 1]
        E1 --> F2[Past Grader 2]
        E1 --> F3[Past Grader 3]
        F1 --> G1[Past Grading Judge]
        F2 --> G1
        F3 --> G1
        G1 --> H1[Past Grade]
        H1 --> I1[Past Attribution Agent]
        H1 --> J1[Past Certainty Agent]
        I1 --> K1[Past Attribution Score]
        J1 --> L1[Past Certainty Score]
    end
    
    subgraph "Current Events Processing"
        D --> E2[Current Event Context]
        E2 --> F4[Current Grader 1]
        E2 --> F5[Current Grader 2]
        E2 --> F6[Current Grader 3]
        F4 --> G2[Current Grading Judge]
        F5 --> G2
        F6 --> G2
        G2 --> H2[Current Grade]
        H2 --> I2[Current Attribution Agent]
        H2 --> J2[Current Certainty Agent]
        I2 --> K2[Current Attribution Score]
        J2 --> L2[Current Certainty Score]
    end
    
    subgraph "Final Aggregation"
        H1 --> M[Max Grade Selection]
        H2 --> M
        K1 --> N[Max Attribution Selection]
        K2 --> N
        L1 --> O[Max Certainty Selection]
        L2 --> O
        M --> P[Final Event Result]
        N --> P
        O --> P
    end
    
    style C fill:#e8f5e8
    style D fill:#e1f5fe
    style E1 fill:#e8f5e8
    style E2 fill:#e1f5fe
    style M fill:#fff3e0
    style N fill:#fff3e0
    style O fill:#fff3e0
```

## 6. Evidence Flow Architecture

Shows how evidence is collected and propagated through the system.

```mermaid
graph TD
    A[Clinical Note] --> B[Evidence Extraction Pipeline]
    
    subgraph "Identification Evidence"
        B --> C1[Identifier 1 Evidence]
        B --> C2[Identifier 2 Evidence]
        B --> C3[Identifier 3 Evidence]
        C1 --> D[Judge Aggregated Evidence]
        C2 --> D
        C3 --> D
    end
    
    subgraph "Grading Evidence"
        D --> E1[Past Grading Evidence]
        D --> E2[Current Grading Evidence]
        E1 --> F1[Past Judge Evidence]
        E2 --> F2[Current Judge Evidence]
    end
    
    subgraph "Attribution Evidence"
        F1 --> G1[Past Attribution Evidence]
        F2 --> G2[Current Attribution Evidence]
    end
    
    subgraph "Certainty Evidence"
        F1 --> H1[Past Certainty Evidence]
        F2 --> H2[Current Certainty Evidence]
    end
    
    subgraph "Structured Evidence Output"
        G1 --> I[Evidence Dictionary]
        G2 --> I
        H1 --> I
        H2 --> I
        D --> I
        F1 --> I
        F2 --> I
        I --> J[Final Evidence Structure]
    end
    
    J --> K[User Overview Generation]
    
    style D fill:#fff3e0
    style F1 fill:#e8f5e8
    style F2 fill:#e1f5fe
    style I fill:#fce4ec
    style J fill:#f3e5f5
```

## 7. Iteration and Quality Control

Shows the meta-judge feedback loop for quality improvement.

```mermaid
graph TD
    A[Initial Processing] --> B[Meta-Judge Evaluation]
    
    subgraph "Quality Assessment"
        B --> C{Quality Check}
        C -->|Satisfactory| D[Generate User Overview]
        C -->|Needs Improvement| E[Iteration Required]
    end
    
    subgraph "Improvement Loop"
        E --> F[Feedback Analysis]
        F --> G[Parameter Adjustment]
        G --> H[Retry Processing]
        H --> B
    end
    
    subgraph "Iteration Control"
        I[Max Iterations Check]
        E --> I
        I -->|Under Limit| F
        I -->|At Limit| J[Force Completion]
        J --> D
    end
    
    D --> K[Final Result with Overview]
    
    style B fill:#fce4ec
    style C fill:#fff3e0
    style E fill:#ffebee
    style I fill:#e8f5e8
    style K fill:#e3f2fd
```

## 8. Agent Configuration Matrix

Shows the different agent types and their configurable parameters.

```mermaid
graph TD
    A[Agent Configuration] --> B[Agent Types]
    
    subgraph "Core Agents"
        B --> C1[Temporal Identifiers]
        B --> C2[Past Graders]
        B --> C3[Current Graders]
        B --> C4[Attribution Detector]
        B --> C5[Certainty Assessor]
    end
    
    subgraph "Judge Agents"
        B --> D1[Identification Judge]
        B --> D2[Past Grading Judge]
        B --> D3[Current Grading Judge]
        B --> D4[Meta Judge]
    end
    
    subgraph "Support Agents"
        B --> E1[Overview Generator]
    end
    
    subgraph "Configuration Parameters"
        F1[Model Type]
        F2[Agent Count]
        F3[Prompt Variant]
        F4[Temperature]
        F5[Max Tokens]
    end
    
    C1 --> F1
    C1 --> F2
    C2 --> F2
    C3 --> F2
    D1 --> F3
    D2 --> F3
    D3 --> F3
    
    style C1 fill:#e1f5fe
    style C2 fill:#e8f5e8
    style C3 fill:#e8f5e8
    style D1 fill:#fff3e0
    style D2 fill:#fff3e0
    style D3 fill:#fff3e0
    style D4 fill:#fce4ec
```

## Architecture Comparison Summary

| Variant | Identifier Agents | Judge Usage | Self-Consistency | Use Case |
|---------|------------------|-------------|------------------|----------|
| **Default** | 3 agents | Full judges | High | Production quality |
| **ablation_single** | 1 agent | Full judges | None | Speed optimization |
| **ablation_no_judge** | 3 agents | No judges | Medium | Cost optimization |
| **Single + No Judge** | 1 agent | No judges | None | Minimal baseline |

## Key Benefits by Variant

### Multi-Agent with Judges (Default)
- **Highest accuracy** through consensus
- **Quality control** via judge evaluation
- **Robust evidence** collection
- **Best for production** use cases

### Single Agent
- **Faster processing** (fewer API calls)
- **Lower cost** (reduced token usage)
- **Simpler debugging** (single decision path)
- **Good for development** and testing

### No Judge
- **Reduced complexity** (no consensus needed)
- **Lower latency** (direct results)
- **Cost savings** (fewer judge calls)
- **Suitable for batch** processing

### Temporal Separation
- **Context-specific** analysis
- **Improved accuracy** for time-sensitive events
- **Separate evidence** tracking
- **Clinical relevance** preservation 