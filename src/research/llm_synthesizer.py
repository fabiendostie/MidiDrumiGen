"""LLM-based synthesis of producer style parameters from web data."""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Try to import Anthropic, but make it optional
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed, Claude API unavailable")

# Try to import OpenAI as backup
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed, GPT-4 backup unavailable")

# Try to import Google Gemini as backup
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini package not installed, Gemini backup unavailable")


class StyleSynthesizer:
    """
    Use LLM to analyze scraped data and extract drum programming style parameters.

    Supports Claude (primary) and GPT-4 (backup).

    Example:
        >>> synthesizer = StyleSynthesizer(claude_api_key="...")
        >>> scraped_data = {...}  # From web scraper
        >>> style_profile = await synthesizer.synthesize_style("Timbaland", scraped_data)
        >>> print(style_profile['tempo_range'])
        [95, 140]
    """

    def __init__(
        self,
        claude_api_key: str | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """
        Initialize style synthesizer.

        Args:
            claude_api_key: Anthropic API key (optional, reads from env)
            openai_api_key: OpenAI API key (backup, optional)
            gemini_api_key: Google Gemini API key (backup, optional)
            model: Claude model to use
        """
        self.model = model

        # Initialize Claude client
        self.claude_client = None
        if ANTHROPIC_AVAILABLE:
            api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.claude_client = Anthropic(api_key=api_key)
                logger.info(f"Claude API initialized with model: {model}")
            else:
                logger.warning("No Claude API key provided")
        else:
            logger.warning("Anthropic package not available")

        # Initialize Gemini client (backup - preferred over OpenAI)
        self.gemini_client = None
        if GEMINI_AVAILABLE:
            api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel("gemini-2.0-flash-exp")
                logger.info("Gemini API initialized (backup)")
        else:
            logger.debug("Google Gemini package not available (optional)")

        # Initialize OpenAI client (backup)
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI API initialized (backup)")
        else:
            logger.debug("OpenAI package not available (optional)")

    def _create_synthesis_prompt(self, producer_name: str, scraped_data: dict[str, Any]) -> str:
        """
        Create prompt for LLM to analyze scraped data.

        Args:
            producer_name: Producer name
            scraped_data: Aggregated data from web scraper

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze the following information about music producer "{producer_name}" and extract their characteristic drum programming style parameters.

SCRAPED DATA:
{json.dumps(scraped_data, indent=2)}

Based on this information, extract and return a JSON object with the following structure:

{{
  "tempo_range": [min_bpm, max_bpm],
  "swing_percentage": 50-75,
  "micro_timing_ms": 0-20,
  "ghost_note_prob": 0.0-1.0,
  "velocity_variation": 0.0-0.3,
  "quantization_grid": "16th" | "32nd" | "triplet",
  "signature_techniques": ["technique1", "technique2", ...],
  "genre_tags": ["genre1", "genre2", ...],
  "complexity_level": 0.0-1.0,
  "description": "brief summary of their drum style"
}}

PARAMETER GUIDELINES:

1. **tempo_range**: Typical BPM range for this producer's work
   - Hip-hop: 80-100
   - Trap: 130-150
   - Rock: 110-140
   - Jazz: 120-240

2. **swing_percentage**: How far from straight quantization
   - 50 = perfectly straight (quantized)
   - 60-62 = J Dilla signature swing
   - 67 = triplet feel
   - 75 = heavy swing

3. **micro_timing_ms**: Timing looseness/tightness
   - 0-5ms = very tight (electronic, trap)
   - 5-10ms = moderate (most hip-hop)
   - 10-20ms = loose, humanized (jazz, boom bap)

4. **ghost_note_prob**: Likelihood of ghost notes (0.0-1.0)
   - 0.0-0.1 = minimal (trap, electronic)
   - 0.2-0.3 = moderate (hip-hop)
   - 0.4-0.6 = heavy (funk, jazz)

5. **velocity_variation**: Dynamic range (0.0-0.3)
   - 0.05-0.1 = tight (trap, electronic)
   - 0.1-0.2 = moderate (hip-hop)
   - 0.2-0.3 = very dynamic (jazz, live feel)

6. **quantization_grid**: Primary note division
   - "16th" = standard hip-hop, rock
   - "32nd" = trap hi-hats, electronic
   - "triplet" = swing, jazz

7. **signature_techniques**: Specific drumming characteristics
   - Examples: "syncopated hi-hats", "kick on 1 and 3", "off-grid snares", "vocal-like percussion"

8. **genre_tags**: Musical genres associated with this producer

9. **complexity_level**: Overall pattern complexity (0.0-1.0)
   - 0.0-0.3 = simple, minimal
   - 0.3-0.6 = moderate complexity
   - 0.6-1.0 = complex, polyrhythmic

10. **description**: 1-2 sentence summary of their drum programming style

IMPORTANT:
- If information is missing, use reasonable defaults based on genre
- Prioritize information from Wikipedia intro and YouTube tutorial transcripts
- Be specific about signature techniques that define this producer's sound
- Return ONLY valid JSON, no additional text

JSON OUTPUT:"""

        return prompt

    async def synthesize_with_claude(
        self, producer_name: str, scraped_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Use Claude API to synthesize style parameters.

        Args:
            producer_name: Producer name
            scraped_data: Scraped web data

        Returns:
            Style parameters dict
        """
        if not self.claude_client:
            raise RuntimeError("Claude API not available")

        prompt = self._create_synthesis_prompt(producer_name, scraped_data)

        try:
            logger.info(f"Calling Claude API for {producer_name}...")

            response = self.claude_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent output
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract JSON from response
            response_text = response.content[0].text.strip()

            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # Parse JSON
            style_params = json.loads(response_text)

            logger.info(f"Successfully synthesized style for {producer_name}")
            return style_params

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Claude response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Invalid JSON in Claude response: {e}")
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def synthesize_with_gpt4(
        self, producer_name: str, scraped_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Use GPT-4 API as backup for style synthesis.

        Args:
            producer_name: Producer name
            scraped_data: Scraped web data

        Returns:
            Style parameters dict
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI API not available")

        prompt = self._create_synthesis_prompt(producer_name, scraped_data)

        try:
            logger.info(f"Calling GPT-4 API for {producer_name}...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            style_params = json.loads(response_text)

            logger.info(f"Successfully synthesized style with GPT-4 for {producer_name}")
            return style_params

        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            raise

    async def synthesize_with_gemini(
        self, producer_name: str, scraped_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Use Gemini API for style synthesis.

        Args:
            producer_name: Producer name
            scraped_data: Scraped web data

        Returns:
            Style parameters dict
        """
        if not self.gemini_client:
            raise RuntimeError("Gemini API not available")

        prompt = self._create_synthesis_prompt(producer_name, scraped_data)

        try:
            logger.info(f"Calling Gemini API for {producer_name}...")

            response = self.gemini_client.generate_content(prompt)
            response_text = response.text.strip()

            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            style_params = json.loads(response_text)

            logger.info(f"Successfully synthesized style with Gemini for {producer_name}")
            return style_params

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def synthesize_style(
        self, producer_name: str, scraped_data: dict[str, Any], use_backup: bool = True
    ) -> dict[str, Any]:
        """
        Synthesize style parameters from scraped data.

        Tries Claude first, then Gemini, then GPT-4 as fallbacks.

        Args:
            producer_name: Producer name
            scraped_data: Aggregated scraped data
            use_backup: Whether to use backup LLMs

        Returns:
            Style parameters dict

        Example:
            >>> synthesizer = StyleSynthesizer()
            >>> style = await synthesizer.synthesize_style("Timbaland", scraped_data)
            >>> print(style['swing_percentage'])
            54.0
        """
        # Try Claude first
        if self.claude_client:
            try:
                return await self.synthesize_with_claude(producer_name, scraped_data)
            except Exception as e:
                logger.warning(f"Claude synthesis failed: {e}")
                if not use_backup:
                    raise

        # Fall back to Gemini (preferred backup)
        if use_backup and self.gemini_client:
            try:
                return await self.synthesize_with_gemini(producer_name, scraped_data)
            except Exception as e:
                logger.warning(f"Gemini synthesis failed: {e}")

        # Fall back to GPT-4 (secondary backup)
        if use_backup and self.openai_client:
            try:
                return await self.synthesize_with_gpt4(producer_name, scraped_data)
            except Exception as e:
                logger.error(f"GPT-4 synthesis also failed: {e}")

        # If all failed or unavailable, return default parameters
        logger.warning(f"LLM synthesis unavailable, using default parameters for {producer_name}")
        return self._create_default_params(producer_name, scraped_data)

    def _create_default_params(
        self, producer_name: str, scraped_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create default style parameters if LLM is unavailable.

        Uses basic heuristics from scraped data.

        Args:
            producer_name: Producer name
            scraped_data: Scraped data

        Returns:
            Default style parameters
        """
        # Extract genres from Wikipedia
        genres = scraped_data.get("wikipedia", {}).get("genres", [])

        # Genre-based defaults
        if any(g in ["hip hop", "hip-hop", "rap"] for g in genres):
            tempo_range = [85, 100]
            swing = 58.0
            quantization = "16th"
        elif any(g in ["trap"] for g in genres):
            tempo_range = [130, 150]
            swing = 52.0
            quantization = "32nd"
        elif any(g in ["jazz"] for g in genres):
            tempo_range = [120, 180]
            swing = 65.0
            quantization = "triplet"
        elif any(g in ["rock", "metal"] for g in genres):
            tempo_range = [110, 140]
            swing = 50.0
            quantization = "16th"
        else:
            # Generic defaults
            tempo_range = [90, 130]
            swing = 55.0
            quantization = "16th"

        return {
            "tempo_range": tempo_range,
            "swing_percentage": swing,
            "micro_timing_ms": 10.0,
            "ghost_note_prob": 0.25,
            "velocity_variation": 0.15,
            "quantization_grid": quantization,
            "signature_techniques": ["default pattern", "standard groove"],
            "genre_tags": genres if genres else ["unknown"],
            "complexity_level": 0.5,
            "description": f"Default parameters based on {genres[0] if genres else 'unknown'} genre",
            "synthesized_with": "default_heuristics",
        }
