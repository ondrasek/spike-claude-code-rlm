# Literary Analysis — Claude Code Workflow

Interactive workflow for analyzing "The Adventures of Sherlock Holmes" using the RLM plugin.

## Prerequisites

1. Fetch the text (if not already downloaded):
   ```bash
   bash fetch_text.sh
   ```
2. RLM plugin installed in Claude Code (see [plugin docs](../../plugin/README.md))

## Step 1: Theme Analysis

Ask Claude Code:

```
/rlm:rlm examples/04-literary-analysis/sherlock_holmes.txt What are the major themes across the stories in this collection? Identify recurring motifs and how they evolve.
```

## Step 2: Deductive Reasoning Deep-Dive

This query exercises recursive `llm_query()` — the LLM can make sub-queries to itself while processing:

```
/rlm:rlm examples/04-literary-analysis/sherlock_holmes.txt For each story, identify the key deductive reasoning chain Holmes uses to solve the case. Compare the logical methods across stories — does Holmes rely more on observation, elimination, or inference?
```

## Step 3: Character Relationship Mapping

```
/rlm:rlm examples/04-literary-analysis/sherlock_holmes.txt Map the relationships between recurring characters (Holmes, Watson, Irene Adler, Lestrade, Mrs. Hudson, Moriarty). For each pair that interacts, describe the nature of their relationship and how it develops.
```

## What to Expect

- **Step 1** demonstrates basic large-text analysis — the REPL will chunk and search the text
- **Step 2** may trigger `llm_query()` recursion as the LLM breaks down each story's reasoning chain
- **Step 3** shows structured extraction from unstructured narrative text
