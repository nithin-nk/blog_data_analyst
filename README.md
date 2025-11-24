# blog_data_analyst
AI Agent data analyst who do research, collect and fine tune by blog post

Steps to create a blog post
    1. The user will give the blog idea and outline of the blog in the form of questions . Format will be in YAML file
            For Example
            Topic:
                How to build memory for AI agents using the open source framework mem0?
            Outline:
                1. What is memory for AI agents?
                2. Why AI agents need memory with an example?
                3. Whay we cannot use traditional databases for memory for AI agents?
                4. Waht is the architecture for mem0 memory?
                    Mermaid: Archetural digram 
                5. Why we need vector database for storing memory for mem0?
                6. Why we need graph database for mem0?
                7. How to extract custom memories using mem0?
                    Code: custom prompt
                8. Create different memory layers for different sub-agents in mem0? 
                9. How to incorporate extracted memory in an AI agent prompt with an example?
                    Code: Prompt for a sample AI agent
                9. How will the memory layer improve the AI agents?
                10. Conclusion
    2. Get web URLs by doing a Google search of different topics in the outline - https://github.com/browser-use/browser-use
    3. Extract the contents from the web pages - https://github.com/unclecode/crawl4ai
    4. AI Agent generates the content for all the subtopics with citation
    5. AI Agent generate the combined blog post for all the topic
    6. AI generate the catchy title, tags and search description
    7. The AI agent generates the code and mermaid diagram
    7. AI agent to refine the blog post, make it a cohesive unit 
    8. Do SEO
        Keyword density optimization
        Internal/external linking suggestions
        Meta description optimization
        Readability scoring (Flesch-Kincaid, etc.)
        Header hierarchy validation (H1→H2→H3)
    9. Check the quality of the blog post out of 10 with reason for improvement with two LLM
    10. Go to step 7 if score is less than 7 to implement the feedback , with a limit for itration
    11. If the score is above 8, generate an image for the blog post using nano banana
    12. Convert the blog post from md to html
    13. Save the output in an HTML file
