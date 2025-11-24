# Blog Data Analyst

AI Agent data analyst who does research, collects data, and fine-tunes content for blog posts.

## Steps to Create a Blog Post

1. **User Input**: The user provides the blog idea and outline in YAML format with questions.
   
   **Example:**
   ```yaml
   Topic: How to build memory for AI agents using the open source framework mem0?
   
   Outline:
     1. What is memory for AI agents?
     2. Why AI agents need memory with an example?
     3. Why we cannot use traditional databases for memory for AI agents?
     4. What is the architecture for mem0 memory?
        - Mermaid: Architectural diagram 
     5. Why we need vector database for storing memory for mem0?
     6. Why we need graph database for mem0?
     7. How to extract custom memories using mem0?
        - Code: custom prompt
     8. Create different memory layers for different sub-agents in mem0?
     9. How to incorporate extracted memory in an AI agent prompt with an example?
        - Code: Prompt for a sample AI agent
     10. How will the memory layer improve the AI agents?
     11. Conclusion
   ```

2. **Web Research**: Get web URLs by performing a Google search for different topics in the outline
   - Tool: [browser-use](https://github.com/browser-use/browser-use)

3. **Content Extraction**: Extract the contents from the web pages
   - Tool: [crawl4ai](https://github.com/unclecode/crawl4ai)

4. **Content Generation**: AI Agent generates content for all subtopics with citations

5. **Blog Compilation**: AI Agent generates the combined blog post for all topics

6. **Metadata Generation**: AI generates catchy title, tags, and search description

7. **Assets Creation**: AI agent generates code snippets and mermaid diagrams

8. **Content Refinement**: AI agent refines the blog post to make it a cohesive unit

9. **SEO Optimization**:
   - Keyword density optimization
   - Internal/external linking suggestions
   - Meta description optimization
   - Readability scoring (Flesch-Kincaid, etc.)
   - Header hierarchy validation (H1→H2→H3)

10. **Quality Check**: Check the quality of the blog post (score out of 10) with reasons for improvement using two LLMs

11. **Iteration**: If score is less than 7, go back to step 8 to implement feedback (with iteration limit)

12. **Image Generation**: If score is above 8, generate an image for the blog post using Banana

13. **Format Conversion**: Convert the blog post from Markdown to HTML

14. **Save Output**: Save the final output as an HTML file
