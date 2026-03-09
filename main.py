import asyncio
import os
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

async def get_location_info(task: str) -> str:
    """Get location information using AI and geocode MCP"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ ERROR: GEMINI_API_KEY missing in .env file"

    client = genai.Client(api_key=api_key)

    server_params = StdioServerParameters(
        command="uvx",
        args=["geocode-mcp"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            try:
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=task,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        tools=[session],
                        system_instruction="""
                        You are a simple travel guide.
                        DO NOT provide GPS coordinates (latitude/longitude).
                        Instead, provide the full address and a simple description of the location.
                        Tell the user which city, area, or province the place is in.
                        Keep it conversational and easy to understand.
                        """
                    ),
                )
                return response.text

            except Exception as e:
                return f"❌ Error: {e}"

def process_query(query: str) -> str:
    """Sync wrapper for async function"""
    if not query.strip():
        return "⚠️ Please enter a location query"
    
    return asyncio.run(get_location_info(query))

def main():
    """Launch the Gradio UI"""
    print("🚀 Starting Map App UI...\n")

    with gr.Blocks(
        title="🗺️ Map Location Finder",
    ) as demo:
        gr.Markdown("# 🗺️ Map Location Finder")
        gr.Markdown("Ask about any location and get detailed information!")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Enter your query:")
                query_input = gr.Textbox(
                    label="Location Query",
                    placeholder="e.g., Where is Minar-e-Pakistan located?",
                    lines=3,
                )
                submit_btn = gr.Button("🔍 Find Location", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
                
            with gr.Column(scale=2):
                gr.Markdown("### Result:")
                output = gr.Textbox(
                    label="Location Information",
                    lines=10,
                )
        
        gr.Markdown("---")
        gr.Markdown("### 💡 Example Queries:")
        gr.Examples(
            examples=[
                "Where is Minar-e-Pakistan located?",
                "Tell me about the Eiffel Tower",
                "Where is the Great Wall of China?",
                "What can you tell me about Statue of Liberty?",
                "Where is Taj Mahal located?",
            ],
            inputs=query_input,
        )
        
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=query_input,
            outputs=output,
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[query_input, output],
        )
        
        # Allow Enter key to submit
        query_input.submit(
            fn=process_query,
            inputs=query_input,
            outputs=output,
        )
    
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())

if __name__ == "__main__":
    main()