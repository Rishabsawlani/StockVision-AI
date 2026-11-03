
import os
import json
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

PRIMARY_MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)


def generate_stock_analysis(stock_metrics):
    """
    Generate historical stock analysis using Gemini.
    """

    prompt = f"""
    You are a professional financial data analyst.

    Analyze the following historical stock metrics:

    {json.dumps(stock_metrics, indent=4)}

    Structure your response using these five sections:

    ## 1. Historical Performance
    Explain cumulative return and overall historical performance.

    ## 2. Key Positives
    Identify genuinely positive observations supported by the data.
    If there are no meaningful positives, state that clearly.

    ## 3. Risk Factors
    Explain volatility and maximum drawdown in practical terms.

    ## 4. Sector Comparison
    Compare the stock against its tracked sector peers.
    Mention the peer count and differences in percentage points.

    ## 5. Overall Interpretation
    Summarize historical performance and risk profile.

    STRICT INSTRUCTIONS:
    - Use only the supplied metrics.
    - Never invent financial figures.
    - Do not repeat observations across sections.
    - Distinguish historical facts from interpretations.
    - Do not label negative observations as positives.
    - Explain large differences objectively.
    - Mention insufficient data where applicable.
    - Do not predict future prices.
    - Do not provide buy or sell recommendations.
    - Use concise professional language.
    - Format using Markdown.
    """

    models = [PRIMARY_MODEL, FALLBACK_MODEL]

    last_error = None

    for model_name in models:

        try:

            print(f"Trying Gemini model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            print(f"Analysis generated using {model_name}")

            return response.text

        except Exception as error:

            last_error = error

            error_message = str(error)

            # Use fallback only for temporary server errors
            if "503" in error_message or "UNAVAILABLE" in error_message:

                print(
                    f"{model_name} is temporarily unavailable. "
                    "Trying fallback model..."
                )

                continue

            # Do not hide authentication, quota or other errors
            raise

    raise RuntimeError(
        "Both Gemini models are currently unavailable. "
        "Please try generating the analysis again later. "
        f"Last error: {last_error}"
    )
