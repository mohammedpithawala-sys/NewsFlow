import json
import os
import time
from typing import Optional
 
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel
 
from crewai.flow.flow import Flow, listen, start
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
 
from tools.custom_tool import (
    google_trends_tool,
    html_render_tool,
    unsplash_search_tool,
)
 
load_dotenv()
 
# ─────────────────────────────────────────────────────────────────────────────
# LiteLLM settings
# ─────────────────────────────────────────────────────────────────────────────
 
litellm.num_retries = 5
litellm.request_timeout = 120
litellm.retry_after = 60
 
MISTRAL_KEY   = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "mistral/mistral-small-latest"
 
# ─────────────────────────────────────────────────────────────────────────────
# Shared tool instances
# ─────────────────────────────────────────────────────────────────────────────
 
serper_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
 
# ─────────────────────────────────────────────────────────────────────────────
# Direct LLM helper — 1 API call, no crew overhead
# ─────────────────────────────────────────────────────────────────────────────
 
def call_llm(prompt: str, max_tokens: int = 512, retries: int = 5, wait: int = 90) -> str:
    for attempt in range(retries):
        try:
            response = litellm.completion(
                model=MISTRAL_MODEL,
                api_key=MISTRAL_KEY,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            is_rate = any(x in err for x in ["429", "rate_limit", "rate limit", "1300", "RateLimit"])
            if is_rate and attempt < retries - 1:
                print(f"\n  [Rate limit] Waiting {wait}s before retry {attempt + 1}/{retries}...\n")
                time.sleep(wait)
            else:
                raise
 
# ─────────────────────────────────────────────────────────────────────────────
# Flow State
# ─────────────────────────────────────────────────────────────────────────────
 
class NewsFlowState(BaseModel):
    topics: list[str] = []
    raw_articles: dict[str, str] = {}
    raw_images: dict[str, list[str]] = {}
    edited_articles: dict[str, str] = {}
    output_path: str = ""
    error: Optional[str] = None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# NewsFlow
# ─────────────────────────────────────────────────────────────────────────────
 
class NewsFlow(Flow[NewsFlowState]):
 
    # ── Stage 1 ── Trend discovery — 1 LLM call total ────────────────────────
 
    @start()
    def discover_trends(self):
        print("\n" + "=" * 60)
        print("  NEWSFLOW  |  Stage 1 - Trend Discovery")
        print("=" * 60 + "\n")
 
        # Fetch trends from multiple regions for global coverage
        print("  Fetching global trends...")
        all_trends = []
        for region in ["US", "GB", "IN", "DE", "FR", "JP", "BR", "AU"]:
            try:
                result = google_trends_tool.run(region)
                if result and "error" not in result.lower():
                    all_trends.append(f"[{region}]: {result}")
                    print(f"  {region}: OK")
            except Exception as e:
                print(f"  {region}: failed — {e}")
 
        raw_trends = " | ".join(all_trends) if all_trends else "No trends fetched"
        print(f"  Combined trends length: {len(raw_trends)} chars")
 
        # One single LLM call to pick the best 5
        print("  Picking top 5 global topics (1 LLM call)...")
        prompt = (
            f"Today's trending topics from around the world:\n{raw_trends[:2000]}\n\n"
            "Pick the 5 most newsworthy topics that have GLOBAL significance. "
            "Make sure to cover DIFFERENT regions — not all from the same country. "
            "Skip celebrity gossip, sports scores, and local entertainment. "
            "Prefer topics about politics, economy, war, climate, science, or technology. "
            "Return ONLY a comma-separated list of 5 topic names. "
            "No numbers, no explanation, nothing else.\n"
            "Example: Gaza ceasefire talks, India election results, EU energy crisis, "
            "Global inflation, Amazon deforestation"
        )
        result = call_llm(prompt, max_tokens=80)
        topics = [t.strip() for t in result.split(",") if t.strip()]
        self.state.topics = topics[:5]
        print(f"\n  [Stage 1 done] Topics: {self.state.topics}\n")
        return self.state.topics
 
    # ── Stage 2 ── News scraping — pure Python, zero LLM calls ───────────────
 
    @listen(discover_trends)
    def scrape_news(self, topics: list[str]):
        print("\n" + "=" * 60)
        print("  NEWSFLOW  |  Stage 2 - News Scraping (no LLM)")
        print("=" * 60 + "\n")
 
        for topic in topics:
            print(f"  Scraping: {topic}")
            combined = ""
 
            try:
                # Search for articles — Serper returns snippets + links
                search_result = serper_tool._run(search_query=f"{topic} news latest")
                search_text = str(search_result)
                print(f"  Serper returned {len(search_text)} chars")
 
                # Use the search snippets directly as source material
                # (Serper already includes title + snippet per result)
                combined = search_text[:3000]
 
                # Also try to scrape the first URL found in results
                import re
                urls_found = re.findall(r'https?://[^\s\'"<>]+', search_text)
                urls_found = [u.rstrip(".,)") for u in urls_found][:3]
 
                for url in urls_found:
                    try:
                        print(f"  Scraping: {url[:60]}...")
                        text = scrape_tool.run(url)
                        scraped = str(text)[:1000]
                        if len(scraped) > 100:  # only add if meaningful content
                            combined += f"\n\n--- {url} ---\n{scraped}"
                    except Exception as e:
                        print(f"    Scrape failed: {e}")
 
            except Exception as e:
                print(f"  Search failed for {topic}: {e}")
                combined = f"Recent news about {topic}."
 
            self.state.raw_articles[topic] = combined
            print(f"  [Got {len(combined)} chars for: {topic}]")
 
        print(f"\n  [Stage 2 done] Scraped {len(self.state.raw_articles)} topics\n")
        return dict(self.state.raw_articles)
 
    # ── Stage 3 ── Image fetching — pure Python, zero LLM calls ──────────────
 
    @listen(scrape_news)
    def scrape_images(self, raw_articles: dict):
        print("\n" + "=" * 60)
        print("  NEWSFLOW  |  Stage 3 - Image Fetching (no LLM)")
        print("=" * 60 + "\n")
 
        import requests
 
        unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
 
        for topic in self.state.topics:
            print(f"  Images for: {topic}")
            keyword = " ".join(topic.lower().split()[:3])
            urls = []
 
            # Try Unsplash API directly
            if unsplash_key:
                try:
                    resp = requests.get(
                        "https://api.unsplash.com/search/photos",
                        params={"query": keyword, "per_page": 3, "orientation": "landscape"},
                        headers={"Authorization": f"Client-ID {unsplash_key}"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    results = resp.json().get("results", [])
                    urls = [r["urls"]["regular"] for r in results]
                    print(f"  Unsplash: {len(urls)} image(s)")
                except Exception as e:
                    print(f"  Unsplash failed: {e}")
            else:
                print("  No UNSPLASH_ACCESS_KEY set — skipping Unsplash")
 
            # Try Serper image search as fallback
            if len(urls) < 3:
                try:
                    serper_key = os.getenv("SERPER_API_KEY", "").strip()
                    resp = requests.post(
                        "https://google.serper.dev/images",
                        headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                        json={"q": f"{topic} news", "num": 5},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    image_results = resp.json().get("images", [])
                    serper_urls = [img["imageUrl"] for img in image_results if img.get("imageUrl")]
                    urls.extend(serper_urls)
                    print(f"  Serper images: {len(serper_urls)} image(s)")
                except Exception as e:
                    print(f"  Serper images failed: {e}")
 
            self.state.raw_images[topic] = urls[:3]
            print(f"  [Got {len(self.state.raw_images[topic])} images for: {topic}]")
 
        print(f"\n  [Stage 3 done]\n")
        return dict(self.state.raw_images)
 
    # ── Stage 4 ── Content editing — 1 LLM call per topic ────────────────────
 
    @listen(scrape_images)
    def edit_articles(self, raw_images: dict):
        print("\n" + "=" * 60)
        print("  NEWSFLOW  |  Stage 4 - Content Editing (1 call/topic)")
        print("=" * 60 + "\n")
 
        for topic in self.state.topics:
            raw = self.state.raw_articles.get(topic, "")
            if not raw:
                self.state.edited_articles[topic] = "No source material available."
                continue
 
            print(f"  Editing: {topic}")
 
            prompt = (
                f"You are a professional news editor. "
                f"Write a concise factual news article about: {topic}\n\n"
                f"Source material:\n"
                f"{raw[:1500]}\n\n"
                "Rules:\n"
                "- 150-200 words\n"
                "- Neutral AP/Reuters wire style\n"
                "- Strong opening sentence, 2 body paragraphs, brief closing\n"
                "- Plain prose only — no bullet points, no markdown, no headers\n"
                "- Only use facts from the source material above\n\n"
                "Return ONLY the article text."
            )
 
            try:
                article = call_llm(prompt, max_tokens=400)
                if article and len(article) > 50:
                    self.state.edited_articles[topic] = article
                    print(f"  [Edited {len(article)} chars for: {topic}]")
                else:
                    print(f"  LLM returned empty/short response, using raw content")
                    self.state.edited_articles[topic] = raw[:800]
            except Exception as e:
                print(f"  Edit failed for {topic}: {e}")
                # Use raw search snippets as fallback so page is not empty
                self.state.edited_articles[topic] = raw[:800]
 
            time.sleep(30)  # 30s pause between topics
 
        print(f"\n  [Stage 4 done] Edited {len(self.state.edited_articles)} articles\n")
        print(f"  DEBUG Stage 4 state: {list(self.state.edited_articles.keys())}")
        for t, a in self.state.edited_articles.items():
            print(f"  [{t}]: {len(a)} chars — {a[:80]}")
 
        # Return a plain dict — Pydantic model dicts can be stale when passed via @listen
        return dict(self.state.edited_articles)
 
    # ── Stage 5 ── HTML render — pure Python, zero LLM calls ─────────────────
 
    @listen(edit_articles)
    def render_digest(self, edited_articles: dict):
        print("\n" + "=" * 60)
        print("  NEWSFLOW  |  Stage 5 - HTML Render (no LLM)")
        print("=" * 60 + "\n")
 
        # Use the parameter passed by @listen — it is the return value of edit_articles
        # self.state.edited_articles may be a stale Pydantic copy
        articles = edited_articles if edited_articles else dict(self.state.edited_articles)
        topics   = list(self.state.topics)
        images   = dict(self.state.raw_images)
 
        print(f"  topics: {topics}")
        print(f"  articles keys: {list(articles.keys())}")
        print(f"  images keys: {list(images.keys())}")
 
        payload = {
            "topics":   topics,
            "articles": articles,
            "images":   images,
        }
 
        output_path = html_render_tool.run(json.dumps(payload, ensure_ascii=False))
        self.state.output_path = output_path.strip()
        print(f"\n  [Stage 5 done] Digest: {self.state.output_path}\n")
        return self.state.output_path
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    print("\n" + "#" * 60)
    print("  NEWSFLOW - CrewAI Trending News Pipeline")
    print("#" * 60 + "\n")
 
    flow = NewsFlow()
    flow.kickoff()
 
    print("\n" + "#" * 60)
    print("  NEWSFLOW COMPLETE")
    print(f"  Output : {flow.state.output_path}")
    print(f"  Topics : {flow.state.topics}")
    print("#" * 60 + "\n")
 
    if flow.state.output_path and os.path.exists(flow.state.output_path):
        print(f"  Open: file://{os.path.abspath(flow.state.output_path)}\n")
 
 
if __name__ == "__main__":
    main()