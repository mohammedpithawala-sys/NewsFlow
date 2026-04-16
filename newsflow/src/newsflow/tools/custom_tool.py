"""
custom_tools.py
===============
All custom @tool definitions for the NewsFlow pipeline.

Tools defined here:
  - google_trends_tool      : Fetches trending topics from Google Trends RSS (no API key)
  - unsplash_search_tool    : Searches Unsplash for royalty-free images (UNSPLASH_ACCESS_KEY)
  - html_render_tool        : Renders the final HTML news digest and writes it to disk
"""

import os
import json
import datetime
import requests
import xml.etree.ElementTree as ET

from crewai.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Google Trends Tool
#     No API key required. Reads from Google's public daily trends RSS feed.
# ─────────────────────────────────────────────────────────────────────────────

@tool("GoogleTrendsTool")
def google_trends_tool(region: str = "US") -> str:
    """
    Fetches today's top trending search topics from Google Trends.

    Args:
        region: Two-letter country code, e.g. 'US', 'IN', 'GB'. Defaults to 'US'.

    Returns:
        A comma-separated string of up to 10 trending topic names.
        Returns an error message string if the feed cannot be reached.
    """
    feed_url = (
        f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={region}"
    )
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsFlowBot/1.0)"}

    try:
        resp = requests.get(feed_url, headers=headers, timeout=12)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles = [
            item.find("title").text
            for item in root.iter("item")
            if item.find("title") is not None
        ]
        if not titles:
            return "No trending topics found in feed."
        return ", ".join(titles[:10])
    except requests.RequestException as e:
        return f"Network error fetching Google Trends: {e}"
    except ET.ParseError as e:
        return f"XML parse error from Google Trends feed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Unsplash Image Search Tool
#     Requires UNSPLASH_ACCESS_KEY environment variable (free at unsplash.com/developers).
#     Falls back gracefully with a clear message when the key is missing.
# ─────────────────────────────────────────────────────────────────────────────

@tool("UnsplashSearchTool")
def unsplash_search_tool(query: str) -> str:
    """
    Searches Unsplash for high-quality, royalty-free images matching the query.

    Args:
        query: A short keyword phrase describing the image needed,
               e.g. 'artificial intelligence chip' or 'flood disaster relief'.

    Returns:
        A comma-separated string of up to 3 landscape image URLs (regular size).
        Returns an error/fallback message string if the search fails.

    Requires:
        UNSPLASH_ACCESS_KEY environment variable.
    """
    access_key = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
    if not access_key:
        return (
            "UNSPLASH_ACCESS_KEY not set — skipping Unsplash. "
            "Images will be sourced from web search instead."
        )

    api_url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "per_page": 3,
        "orientation": "landscape",
        "content_filter": "high",
    }
    headers = {"Authorization": f"Client-ID {access_key}"}

    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=12)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return f"No Unsplash images found for query: '{query}'"
        urls = [r["urls"]["regular"] for r in results]
        return ", ".join(urls)
    except requests.RequestException as e:
        return f"Unsplash API error: {e}"
    except (KeyError, ValueError) as e:
        return f"Unsplash response parse error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HTML Render Tool
#     Accepts a JSON payload, builds a newspaper-style HTML digest,
#     writes it to output/digest.html, and returns the file path.
# ─────────────────────────────────────────────────────────────────────────────

@tool("HTMLRenderTool")
def html_render_tool(payload_json: str) -> str:
    """
    Renders a complete HTML news digest from the provided data and saves it to disk.

    Args:
        payload_json: A JSON string with the following structure:
            {
                "topics":   ["Topic A", "Topic B", ...],
                "articles": {"Topic A": "article text...", ...},
                "images":   {"Topic A": ["url1", "url2", "url3"], ...}
            }

    Returns:
        The file path of the written HTML file (e.g. 'output/digest.html'),
        or an error message string if rendering fails.
    """
    # ── Parse payload ─────────────────────────────────────────────────────────
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return f"JSON parse error — could not read payload: {e}"

    topics   = data.get("topics", [])
    articles = data.get("articles", {})
    images   = data.get("images", {})

    if not topics:
        return "Payload contained no topics — nothing to render."

    today = datetime.date.today().strftime("%B %d, %Y")

    # ── Build article cards ───────────────────────────────────────────────────
    cards_html = ""
    for topic in topics:
        article_text = articles.get(topic, "Article content not available.")
        img_urls     = images.get(topic, [])

        # Escape any stray HTML characters in the article text
        safe_text = (
            article_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        hero_html = ""
        if img_urls:
            hero_html = (
                f'<img src="{img_urls[0]}" '
                f'alt="Image for {topic}" class="hero-img" '
                f'onerror="this.style.display=\'none\'" />'
            )

        thumbs_html = ""
        for url in img_urls[1:3]:
            thumbs_html += (
                f'<img src="{url}" alt="{topic}" class="thumb-img" '
                f'onerror="this.style.display=\'none\'" />'
            )

        cards_html += f"""
    <article class="news-card">
      <div class="card-media">
        {hero_html}
        {'<div class="thumbs">' + thumbs_html + '</div>' if thumbs_html else ''}
      </div>
      <div class="card-body">
        <h2 class="topic-title">{topic}</h2>
        <div class="article-text">{safe_text}</div>
      </div>
    </article>"""

    # ── Full HTML document ────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Trending News Digest — {today}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link
    href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap"
    rel="stylesheet"
  />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --ink:     #17120d;
      --paper:   #f4efe6;
      --surface: #ede7db;
      --rule:    #c5b49a;
      --accent:  #8c1c1c;
      --muted:   #6b5d4e;
      --light:   #a89880;
    }}

    body {{
      background: var(--paper);
      color: var(--ink);
      font-family: 'Source Serif 4', Georgia, serif;
      min-height: 100vh;
    }}

    /* ── masthead ── */
    header {{
      padding: 2.4rem 4rem 1.6rem;
      border-bottom: 3px double var(--rule);
      display: flex;
      align-items: baseline;
      gap: 2.4rem;
      flex-wrap: wrap;
    }}

    .masthead {{
      font-family: 'Playfair Display', Georgia, serif;
      font-size: clamp(2.2rem, 5vw, 4rem);
      font-weight: 900;
      letter-spacing: -0.03em;
      line-height: 1;
      color: var(--ink);
    }}

    .masthead span {{
      color: var(--accent);
    }}

    .dateline {{
      font-size: 0.78rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding-left: 1.4rem;
      border-left: 1px solid var(--rule);
    }}

    /* ── topic nav ── */
    .topic-nav {{
      background: var(--surface);
      border-bottom: 1px solid var(--rule);
      padding: 0.6rem 4rem;
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
    }}

    .topic-pill {{
      font-size: 0.72rem;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      padding: 0.25rem 0.8rem;
      border: 1px solid var(--rule);
      color: var(--muted);
      text-decoration: none;
      border-radius: 2px;
      transition: background 0.15s, color 0.15s;
    }}

    .topic-pill:hover {{
      background: var(--accent);
      color: var(--paper);
      border-color: var(--accent);
    }}

    /* ── digest grid ── */
    .digest {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 3rem 3rem 5rem;
      display: flex;
      flex-direction: column;
      gap: 4rem;
    }}

    /* ── individual card ── */
    .news-card {{
      display: grid;
      grid-template-columns: minmax(260px, 2fr) 3fr;
      gap: 2.4rem;
      padding-bottom: 4rem;
      border-bottom: 1px solid var(--rule);
    }}

    .news-card:last-child {{
      border-bottom: none;
      padding-bottom: 0;
    }}

    .card-media {{
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}

    .hero-img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border: 1px solid var(--rule);
      display: block;
    }}

    .thumbs {{
      display: flex;
      gap: 0.5rem;
    }}

    .thumb-img {{
      flex: 1;
      height: 72px;
      object-fit: cover;
      border: 1px solid var(--rule);
      opacity: 0.82;
      display: block;
    }}

    .thumb-img:hover {{
      opacity: 1;
    }}

    .card-body {{
      display: flex;
      flex-direction: column;
      gap: 1.1rem;
    }}

    .topic-title {{
      font-family: 'Playfair Display', Georgia, serif;
      font-size: clamp(1.4rem, 2.6vw, 2.1rem);
      font-weight: 700;
      line-height: 1.18;
      color: var(--ink);
      padding-left: 1rem;
      border-left: 4px solid var(--accent);
    }}

    .article-text {{
      font-size: 0.96rem;
      line-height: 1.82;
      color: var(--ink);
      font-weight: 300;
      white-space: pre-wrap;
    }}

    .article-text::first-letter {{
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 3.2rem;
      font-weight: 900;
      float: left;
      line-height: 0.78;
      margin-right: 0.1em;
      color: var(--accent);
    }}

    /* ── footer ── */
    footer {{
      text-align: center;
      padding: 1.6rem;
      font-size: 0.72rem;
      color: var(--light);
      border-top: 1px solid var(--rule);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    /* ── responsive ── */
    @media (max-width: 720px) {{
      header, .topic-nav, .digest {{ padding-left: 1.2rem; padding-right: 1.2rem; }}
      .news-card {{ grid-template-columns: 1fr; gap: 1.4rem; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="masthead">Trending <span>Today</span></div>
  <div class="dateline">AI-curated digest &nbsp;&middot;&nbsp; {today}</div>
</header>

<nav class="topic-nav">
  {''.join(f'<a class="topic-pill" href="#topic-{i}">{t}</a>' for i, t in enumerate(topics))}
</nav>

<div class="digest">
  {''.join(f'<span id="topic-{i}"></span>' + cards_html.split('</article>')[i] + '</article>' if i < len(topics) else '' for i in range(len(topics)))}
</div>

<footer>Generated by NewsFlow &mdash; CrewAI Agentic Pipeline &nbsp;&middot;&nbsp; {today}</footer>

</body>
</html>"""

    # ── Write to disk ─────────────────────────────────────────────────────────
    out_dir  = "output"
    out_path = os.path.join(out_dir, "digest.html")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return out_path
    except OSError as e:
        return f"File write error: {e}"