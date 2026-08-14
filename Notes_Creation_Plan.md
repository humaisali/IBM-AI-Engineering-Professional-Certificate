# General Plan for Creating AI/ML Course Notes

This document outlines the standard operating procedure and generalization plan for analyzing course transcripts and summaries to convert them into clean, structured, and comprehensive Markdown notes.

## 1. Directory & File Structure
- **Location:** Notes should always be saved within the specific Module directory inside the respective Course directory.
- **Naming Convention:** `Module_<Number>_Notes.md` (e.g., `Module_1_Notes.md`).

## 2. Information Extraction Strategy
When processing transcript `.txt` files and summaries, focus on the following core components to ensure notes are visually appealing and easy to digest:
1. **Key Definitions & Concepts:** Identify boldable terms and their simple definitions.
2. **Comparisons & Distinctions:** Look for "versus" or "differences" in the text and extract them into **Markdown Tables** for easy reading.
3. **Processes & Workflows:** Identify sequential steps (e.g., ML lifecycles, data pipelines) and represent them using numbered lists or **Mermaid Flowcharts**.
4. **Categorization:** Group lists of tools, techniques, or algorithms using bullet points and sub-headers.
5. **Practical Examples:** Extract real-world scenarios mentioned in the text (e.g., predicting cancer, recommending beauty products) to contextualize theoretical explanations.
6. **Code Snippets, Lab & Program Files:** ALWAYS consider related `.ipynb` files, Program Files, or other code files when creating notes for a specific concept. Extract actual, well-commented Python code blocks, including data loading, preprocessing, model training, and evaluation metrics, derived directly from these lab and program files.
7. **Primary Focus on ChatGPT Notes:** Study the ChatGPT Notes provided with each lecture as the **primary** source of information and structure. Integrate Transcripts, Summaries, and any other notes around the core insights provided by the ChatGPT Notes to create a detailed, cohesive final document.

## 3. Standard Markdown Template
Use the following structure for every module's notes:

```markdown
# Module [X]: [Module Name]

## 1. [Topic/Video 1 Title]
- **Concept:** Definition.
- **Key Points:** ...

## 2. [Topic/Video 2 Title]
- Include mermaid diagrams if a process is discussed.
...

## [Topic X] Comparisons
| Feature | A | B |
|---|---|---|
| ... | ... | ... |

## Code Examples
\`\`\`python
# Relevant code block demonstrating the concept discussed
\`\`\`
```

## 4. Execution Workflow for Future Modules
1. **List Directory:** Verify the files available in the module folder using `list_dir`.
2. **Read Content:** Read the ChatGPT Notes (prioritize these!), transcripts, summary files, program files, and any other discussions systematically using `view_file` (handle batches if there are many files).
3. **Synthesize (ChatGPT Notes First):** Use the ChatGPT Notes as the foundational structure and primary source of truth. Group related information, eliminating filler words and redundant pleasantries. Keep the output focused purely on academic and practical knowledge by enriching the core ChatGPT Notes with details from the Transcript and Summary.
4. **Format:** Apply the strategy mentioned in Section 2 (Tables, Flowcharts, Code blocks) to the extracted text.
5. **Save:** Write the final `.md` file to the module's folder using `write_to_file` and notify the user of successful completion.
