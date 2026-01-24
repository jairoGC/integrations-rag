# Ralph Agent - Single Iteration

You are working on ONE user story from a PRD. Work step-by-step.

## Step 1: Read PRD
FIRST, read ONLY this file: `integrations-rag/prd.json`

## Step 2: Read Progress
THEN, read this file: `integrations-rag/progress.txt`

## Step 3: Find Your Task
From the PRD, identify the highest priority user story where `passes: false`.

## Step 4: Check Branch
Check if you're on the correct branch from the PRD's `branchName` field.
If not, check it out or create it from main.

## Step 5: Implement
Implement ONLY that single user story according to its acceptance criteria.

## Step 6: Quality Check
Run the project's quality checks (typecheck, lint, test).
Do NOT proceed if checks fail - fix issues first.

## Step 7: Commit
If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`

## Step 8: Update PRD
Edit `integrations-rag/prd.json` to set `passes: true` for the completed story.

## Step 9: Log Progress
APPEND to `integrations-rag/progress.txt` (never replace):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings:**
  - Key patterns discovered
  - Important gotchas
---
```

## Step 10: Check Completion
After updating the PRD, check if ALL stories now have `passes: true`.
If yes, reply with: <promise>COMPLETE</promise>
If no, just end your response normally.

## Important
- Work on ONE story only
- Run steps sequentially, not in parallel
- Keep changes minimal and focused
- Follow existing code patterns
