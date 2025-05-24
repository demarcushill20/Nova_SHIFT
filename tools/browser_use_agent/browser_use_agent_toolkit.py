import asyncio
import os
import textwrap
from typing import Optional, Any

from browser_use import Agent, Browser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from playwright.async_api import BrowserContext

# Load environment variables, potentially redundant if loaded globally, but safe to include
load_dotenv()

# Helper functions adapted from the original script
async def _generate_detailed_report(llm, raw_result, original_task):
    """
    Use Gemini 2.5 Pro Experimental to generate a comprehensive 3000+ word detailed report.
    Internal helper function.
    """
    try:
        prompt = f"""
        You are an expert analyst tasked with creating a comprehensive, detailed report (minimum 3000 words) based on web search results.
        Original task: {original_task}
        Below is the raw result of an AI web search. Please analyze this information and provide an in-depth, comprehensive report that includes:
        1. Executive Summary (concise overview of key findings)
        2. Introduction
           - Background and context of the research topic
           - Objectives and scope of the analysis
           - Methodology used for data collection and analysis
        3. Comprehensive Analysis (the bulk of your report)
           - Detailed examination of all key aspects of the topic
           - Multiple subsections covering different angles/perspectives
           - Integration of facts, statistics, and data points with proper context
           - Thorough explanation of complex concepts
           - Comparison of different viewpoints or approaches when applicable
        4. Key Findings
           - Detailed explanations of the most significant discoveries
           - Analysis of trends, patterns, or notable outliers
           - Interconnections between different findings
        5. Practical Implications
           - Real-world applications of the information
           - Potential impact on relevant stakeholders
           - Actionable insights derived from the analysis
        6. Future Directions
           - Areas requiring further research or exploration
           - Emerging trends or developments to monitor
        7. Conclusion
           - Synthesis of all major points
           - Final assessment and recommendations
        Your report MUST be at least 3000 words and use proper academic/professional structure.
        Present your report in a well-organized format with clear headings, subheadings, and logical flow.
        Use markdown formatting for headings and structure.
        Include relevant examples, case studies, or scenarios to illustrate key points when applicable.
        RAW SEARCH RESULT:
        {raw_result}
        """
        report = await llm.ainvoke(prompt)
        return report.content
    except Exception as e:
        return f"Error generating comprehensive report: {str(e)}\n\nPlease refer to the original result above."

def _format_text(text, wrap_width):
    """Format text with optional line wrapping. Internal helper function."""
    if not isinstance(text, str):
        text = str(text)
    if wrap_width <= 0:
        return text
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if line.strip() == '':
            wrapped_lines.append('')
        else:
            if line.strip().startswith(('#', '-', '*')) or (len(line.strip()) > 1 and line.strip()[0].isdigit() and line.strip()[1] == '.'):
                wrapped_lines.append(line)
            else:
                wrapped = textwrap.fill(line, width=wrap_width)
                wrapped_lines.append(wrapped)
    return '\n'.join(wrapped_lines)

def _get_result_text(result):
    """Extract text content from browser-use agent result. Internal helper function."""
    if isinstance(result, str):
        return result
    try:
        if hasattr(result, 'all_results') and result.all_results:
            last_result = result.all_results[-1]
            if hasattr(last_result, 'extracted_content') and last_result.extracted_content:
                return last_result.extracted_content
        return str(result)
    except:
        return str(result)

async def _generate_summary(llm, raw_result, original_task):
    """
    Use Gemini 2.5 Pro Experimental to generate a concise summary.
    Internal helper function.
    """
    try:
        prompt = f"""
        Based on the following web search results gathered for the task "{original_task}", please provide a concise summary of the key findings.
        Focus on the most important information and present it clearly and briefly, perhaps using bullet points or a short paragraph.
        RAW SEARCH RESULT:
        {raw_result}
        CONCISE SUMMARY:
        """
        summary_result = await llm.ainvoke(prompt)
        return summary_result.content
    except Exception as e:
        return f"Error generating summary: {str(e)}\n\nPlease refer to the original result above."


# The main function exposed as a tool
async def run_browser_use_gemini_task(
    task_description: str,
    temperature: float = 0.0,
    timeout: int = 240,
    generate_report: bool = False,
    summarize: bool = False,
    output_file: Optional[str] = None,
    wrap_width: int = 0 # Default to no wrapping for programmatic use
) -> str:
    """
    Runs a web browsing task using the browser-use library with the Gemini 2.5 Pro model.

    Args:
        task_description (str): The specific task for the agent to perform.
        temperature (float, optional): Temperature setting for the LLM (0.0-1.0). Defaults to 0.0.
        timeout (int, optional): Timeout in seconds for LLM API calls. Defaults to 240.
        generate_report (bool, optional): If True, generates a comprehensive detailed report (3000+ words). Defaults to False.
        summarize (bool, optional): If True, generates a concise summary. Mutually exclusive with generate_report. Defaults to False.
        output_file (Optional[str], optional): Path to save the report or summary as a markdown file. Defaults to None.
        wrap_width (int, optional): Line wrap width for console output (0 for no wrapping). Defaults to 0 (no wrap).

    Returns:
        str: The raw result text, or the generated summary/report if requested.
             Returns an error message string if execution fails.
    """
    print(f"Starting browser-use task: {task_description}")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return "Error: GOOGLE_API_KEY environment variable not set."

    # Initialize the model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-03-25", # Using the preview model as per previous change
            temperature=temperature,
            timeout=timeout,
            google_api_key=google_api_key,
            max_retries=3
        )
    except Exception as e:
        return f"Error initializing LLM: {str(e)}"

    # Initialize browser and context
    browser_instance: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    final_output = ""

    try:
        print("Initializing browser...")
        # Ensure playwright chromium is installed. The user should handle this as per browser-use docs.
        # Consider adding a check or note here if needed.
        browser_instance = Browser() # Use default browser config
        print("Initializing browser context...")
        context = await browser_instance.new_context()
        print("✅ Browser and context initialized.")

        # Create and run the agent
        agent = Agent(
            task=task_description,
            llm=llm,
            browser_context=context
        )

        raw_result_obj = await agent.run()
        raw_result_text = _get_result_text(raw_result_obj)
        final_output = raw_result_text # Default output is the raw result

        print("\n🎯 Task raw result received.")
        # Optional: Log the raw result length or snippet for debugging
        # print(f"Raw result length: {len(raw_result_text)}")

        # --- Report or Summary Generation ---
        output_content = None
        output_type = None

        if generate_report and summarize:
            print("Warning: Both --generate-report and --summarize requested. Prioritizing --generate-report.")
            summarize = False # Ensure mutual exclusivity

        if generate_report:
            print("\n🧠 Generating comprehensive detailed report...")
            output_content = await _generate_detailed_report(llm, raw_result_text, task_description)
            output_type = 'report'
        elif summarize:
            print("\n📝 Generating concise summary...")
            output_content = await _generate_summary(llm, raw_result_text, task_description)
            output_type = 'summary'

        # --- Output Handling ---
        if output_content and output_type:
            final_output = output_content # Update final output if report/summary generated
            if output_file:
                try:
                    output_filename = output_file
                    # Append type if filename doesn't specify clearly
                    if f'_{output_type}.md' not in output_filename.lower() and not output_filename.lower().endswith('.md'):
                        base, ext = os.path.splitext(output_filename)
                        output_filename = f"{base}_{output_type}.md"
                    elif not output_filename.lower().endswith('.md'):
                         output_filename += '.md'

                    output_dir = os.path.dirname(output_filename)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        print(f"Created directory: {output_dir}")

                    with open(output_filename, 'w', encoding='utf-8') as f:
                        f.write(output_content)
                    print(f"\n✅ {output_type.capitalize()} saved to: {output_filename}")
                except Exception as e:
                    print(f"\n❌ Error saving {output_type} to file: {str(e)}")
                    # Continue, as the content is still in final_output
            else:
                 # If no output file, the report/summary is just returned
                 print(f"\n📊 {output_type.capitalize()} generated.")


        print("\n✅ Browser-use task complete.")
        return final_output # Return raw result or generated report/summary

    except Exception as e:
        print(f"\n❌ Error occurred during browser-use task execution: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error during browser-use task execution: {str(e)}" # Return error message

    finally:
        # Cleanup
        if context:
            try:
                print("\nClosing browser context...")
                await context.close()
                print("Context closed.")
            except Exception as ctx_close_err:
                print(f"Error closing context: {ctx_close_err}")
        if browser_instance:
            try:
                print("Closing browser...")
                await browser_instance.close()
                print("Browser closed.")
            except Exception as browser_close_err:
                print(f"Error closing browser: {browser_close_err}")

# Example of how to run this function (for testing purposes)
# async def _test_run():
#     task = "Find the latest news about AI"
#     result = await run_browser_use_gemini_task(task, summarize=True, output_file="ai_news_summary.md")
#     print("\n--- Final Result ---")
#     print(result)
#
# if __name__ == "__main__":
#     # Requires GOOGLE_API_KEY to be set in environment
#     # Also requires playwright install chromium
#     asyncio.run(_test_run())