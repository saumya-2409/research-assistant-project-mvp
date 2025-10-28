"""
¬ INTELLIGENT Research Assistant - Smart Paper Access & Content Extraction
- Intelligently detects which papers are actually accessible
- Fetches full text content from accessible papers
- Provides direct paper links (not just search links)
- Only truly paywalled papers go to "suggested reading"
- Enhanced content extraction and summarization
- Beautiful design preserved
"""

import streamlit as st
import warnings
import os
import logging
import hashlib
import re
import random
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from summarizer import FullPaperSummarizer

from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter
from urllib.parse import quote, urljoin, urlparse

PYPDF_AVAILABLE = False  # Default; not used in main.py
import io  # For BytesIO

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=" § ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import optional libraries
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

#  Ž¨ BEAUTIFUL DESIGN CSS (PRESERVED)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 400;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fafbfc 0%, #f4f6f8 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .css-1d391kg {
        background: white;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.03);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: white;
        padding: 6px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
        border: 1px solid #f1f5f9;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 20px;
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 400;
        transition: all 0.2s ease;
        border: none;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card {
        background: white;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease;
        margin: 6px;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.08);
    }
    
    .metric-number {
        font-size: 2.2rem;
        font-weight: 500;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #64748b;
        font-weight: 400;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .summary-section {
        background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%);
        border-left: 3px solid #667eea;
        padding: 1rem;
        margin: 0.6rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
    }
    
    .summary-section strong {
        color: #1e293b;
        font-weight: 500;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.06);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
    }
    
    .status-full {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #15803d;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #86efac;
    }
    
    .status-abstract {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #a16207;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #facc15;
    }
    
    .status-extracted {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        color: #3730a3;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #a5b4fc;
    }
    
    .cluster-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        border: 1px solid #f1f5f9;
        transition: all 0.2s ease;
        margin: 1rem 0;
    }
    
    .cluster-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.08);
    }
    
    .cluster-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .welcome-step {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 500;
        font-size: 0.9rem;
        margin-right: 0.8rem;
    }
    
    .gap-card {
        background: white;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        border-left: 4px solid #8b5cf6;
    }
    
    .suggested-card {
        background: white;
        border: 1px solid #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
        border-left: 3px solid #f59e0b;
    }
    
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.7rem;
        font-size: 0.95rem;
        font-weight: 400;
        transition: all 0.2s ease;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    .source-status {
        background: #f0f9ff;
        border: 1px solid #38bdf8;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    
    .extraction-status {
        background: #f0f4ff;
        border: 1px solid #6366f1;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'papers_data' not in st.session_state:
    st.session_state.papers_data = []
if 'full_text_papers' not in st.session_state:
    st.session_state.full_text_papers = []
if 'suggested_papers' not in st.session_state:
    st.session_state.suggested_papers = []
if 'clusters' not in st.session_state:
    st.session_state.clusters = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ==================== INTELLIGENT PAPER ACCESS DETECTOR ====================
class IntelligentPaperAccessor:
    """Intelligently detects and accesses papers from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def check_and_extract_paper_content(self, paper: Dict) -> Dict:
        paper = paper.copy()
        access_methods = []
        pdf_url = paper.get('pdf_url')
        if pdf_url:
            access_methods.append(('direct_pdf', pdf_url))
        paper_url = paper.get('url', '')
        if paper_url:
            access_methods.append(('paper_landing', paper_url))  # For indirect links
        semantic_id = paper.get('semantic_scholar_id')
        if semantic_id:
            access_methods.append(('semantic_alternative', f"https://www.semanticscholar.org/paper/{semantic_id}"))
        doi = paper.get('doi', '')
        if doi:
            access_methods.append(('doi_pdf', f"https://doi.org/{doi}"))  # DOI resolver
        extracted_content = None
        working_url = None
        access_type = None
        for method_name, url in access_methods:
            try:
                content = self.try_extract_content(url, method_name)
                if content and len(content) > 200:
                    extracted_content = content[:3000]  # Your length limit
                    working_url = url
                    access_type = method_name
                    break
            except:
                continue
        if extracted_content:
            paper['extracted_content'] = extracted_content
            paper['working_url'] = working_url
            paper['access_type'] = access_type
            paper['pdf_available'] = True
        return paper

    def try_extract_content(self, url: str, method_name: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=10, allow_redirects=True)
            if response.status_code != 200:
                return None
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type:
                content_len = len(response.content)
                if content_len < 500 * 1024:  # Small PDF limit
                    if PYPDF_AVAILABLE:
                        try:
                            reader = PdfReader(io.BytesIO(response.content))
                            text = ''
                            for page in reader.pages[:5]:
                                text += (page.extract_text() or '') + '\n'
                            text = text.strip()[:4000]
                            if len(text) > 200:
                                return text
                        except Exception:
                            pass
                    return f"PDF content available ({content_len / 1024:.0f} KB) - Install pypdf for extraction: pip install pypdf"
                return f"PDF content available for download ({content_len / 1024:.0f} KB)"
            elif 'text/html' in content_type:
                if not BEAUTIFULSOUP_AVAILABLE:
                    return "HTML content (install BeautifulSoup for scraping: pip install beautifulsoup4)"
                soup = BeautifulSoup(response.content, 'html.parser')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                pdf_links = []
                cues = ['pdf', 'download', 'full text', 'access pdf', 'view pdf']
                for a in soup.find_all('a', href=True, limit=20):
                    href = a['href'].lower()
                    text = a.get_text(strip=True).lower()
                    if href.endswith('.pdf') or any(cue in href or cue in text for cue in cues) or 'doi.org' in href:
                        full_href = urljoin(url, a['href'])
                        pdf_links.append(full_href)
                        if len(pdf_links) >= 3:
                            break
                for candidate in pdf_links:
                    try:
                        cand_resp = self.session.get(candidate, timeout=8, allow_redirects=True)
                        if cand_resp.status_code == 200 and 'application/pdf' in cand_resp.headers.get('Content-Type', ''):
                            content_len = len(cand_resp.content)
                            if content_len < 500 * 1024:
                                if PYPDF_AVAILABLE:
                                    try:
                                        reader = PdfReader(io.BytesIO(cand_resp.content))
                                        text = ''
                                        for page in reader.pages[:5]:
                                            text += (page.extract_text() or '') + '\n'
                                        text = text.strip()[:4000]
                                        if len(text) > 200:
                                            return text
                                    except Exception:
                                        pass
                                return f"Indirect PDF found ({content_len / 1024:.0f} KB) - Install pypdf for extraction"
                            return f"Indirect PDF accessed via {method_name} ({content_len / 1024:.0f} KB)"
                    except:
                        continue
                # Fallback HTML extraction
                main_content = soup.find('main') or soup.find('article') or soup.body
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)[:3000]
                    if len(text) > 200:
                        return text
                return soup.get_text(separator=' ', strip=True)[:2000]
            return None
        except requests.exceptions.Timeout:
            return "Timeout accessing content"
        except Exception:
            return None


# ==================== REAL ARXIV FETCHER (SAME AS BEFORE) ====================
class RealArxivFetcher:
    """REAL ArXiv fetcher using arxiv-py library"""
    
    def __init__(self):
        self.rate_limit_delay = 0.5
        self.last_request_time = 0
    
    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        if not ARXIV_AVAILABLE:
            return []
        
        try:
            search_query = f"all:{query}"
            
            search = arxiv.Search(
                query=search_query,
                max_results=min(max_results, 100),
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            count = 0
            
            for result in arxiv.Client().results(search):
                if count % 10 == 0:
                    self._rate_limit()
                
                paper = {
                    'id': result.entry_id,
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'abstract': result.summary[:1000] if result.summary else '',
                    'authors': [author.name for author in result.authors],
                    'published_date': result.published.isoformat(),
                    'updated_date': result.updated.isoformat(),
                    'year': int(result.published.year) if result.published else int(datetime.now().year),
                    'month': result.published.month,
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'doi': result.doi,
                    'journal_ref': result.journal_ref,
                    'source': 'arXiv',
                    'citations': None,
                    'pdf_available': True,
                    'full_text': True
                }
                
                papers.append(paper)
                count += 1
                
                if count >= max_results:
                    break
            
            return papers
            
        except Exception as e:
            st.warning(f"ArXiv fetch error: {str(e)}")
            return []

# ==================== ENHANCED SEMANTIC SCHOLAR FETCHER ====================
# ==================== SEMANTIC SCHOLAR FETCHER WITH API KEY ====================
class SemanticScholarFetcher:
    """Semantic Scholar fetcher using your FREE API key - No rate limiting issues!"""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = "DiHAxNAV2Q9BrBDSeGK2W3r5dqegv4S86gdaD70Z"  # Your free API key
        # API rate limit: 1 request per second
        self.rate_limit_delay = 1.0  # Exactly 1 second per request
        self.last_request_time = 0
        self.max_retries = 3  # Standard retries
    
    def _rate_limit(self):
        """Enforce 1 request/second rate limit with small jitter"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last + random.uniform(0.1, 0.3)  # Jitter for safety
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        """Full search using API key - Reliable and unlimited within rate limits"""
        
        if not query:
            return []
        
        # Use full max_results (no artificial limits)
        max_results = min(max_results, 100)  # API allows up to 100
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()  # 1 second delay
                
                search_url = f"{self.base_url}/paper/search"
                params = {
                    'query': query,
                    'limit': max_results,
                    'fields': 'paperId,title,abstract,authors,year,citationCount,url,venue,openAccessPdf,externalIds,isOpenAccess'
                }
                
                headers = {
                    'User-Agent': 'Research Assistant (B.Tech Project - Educational Use)',
                    'Accept': 'application/json',
                    'x-api-key': self.api_key  # Your API key here - enables 1 req/sec
                }
                
                st.info(f" ” **Semantic Scholar (API)**: Searching... (1 req/sec limit)")
                
                response = requests.get(search_url, params=params, headers=headers, timeout=30)
                
                # Handle API responses
                if response.status_code == 429:
                    wait_time = 2 + (attempt * 2)  # Short backoff for rate limit
                    st.warning(f"âš ï¸ **Semantic Scholar API**: Rate limited! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code != 200:
                    if attempt == self.max_retries - 1:
                        st.error(f"âŒ **Semantic Scholar API**: Failed (Status {response.status_code}). Check API key.")
                        return []
                    continue
                
                # Success
                data = response.json()
                papers = []
                
                for paper_data in data.get('data', []):
                    if not paper_data.get('title'):
                        continue
                    
                    authors = [author.get('name', 'Unknown') for author in paper_data.get('authors', [])[:5]]
                    
                    # PDF and open access detection
                    pdf_url = None
                    pdf_available = False
                    is_open_access = paper_data.get('isOpenAccess', False)
                    
                    open_access_pdf = paper_data.get('openAccessPdf')
                    if open_access_pdf and open_access_pdf.get('url'):
                        pdf_url = open_access_pdf['url']
                        pdf_available = True
                    
                    # External IDs for alternative access
                    external_ids = paper_data.get('externalIds', {})
                    arxiv_id = external_ids.get('ArXiv')
                    doi = external_ids.get('DOI')
                    
                    paper_id = paper_data.get('paperId', '')
                    url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ''
                    
                    alternative_urls = []
                    if arxiv_id:
                        alternative_urls.append(f"https://arxiv.org/abs/{arxiv_id}")
                        if not pdf_url:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                            pdf_available = True
                    if doi:
                        alternative_urls.append(f"https://doi.org/{doi}")
                    
                    paper = {
                        'id': paper_id,
                        'semantic_scholar_id': paper_id,
                        'title': paper_data.get('title', ''),
                        'abstract': paper_data.get('abstract', '')[:1000] if paper_data.get('abstract') else '',
                        'authors': authors,
                        'year': int(paper_data.get('year') or datetime.now().year),
                        'citations': int(paper_data.get('citationCount') or 0),
                        'url': url,
                        'pdf_url': pdf_url,
                        'alternative_urls': alternative_urls,
                        'venue': paper_data.get('venue', ''),
                        'source': 'Semantic Scholar (API)',
                        'pdf_available': pdf_available,
                        'full_text': pdf_available or is_open_access,
                        'arxiv_id': arxiv_id,
                        'doi': doi,
                        'is_open_access': is_open_access
                    }
                    
                    papers.append(paper)
                
                # Sort by year (recent first)
                papers.sort(key=lambda x: x.get('year') or 0, reverse=True)
                
                if papers:
                    st.success(f"âœ… **Semantic Scholar API**: {len(papers)} papers found (full access enabled)")
                else:
                    st.info("â„¹ï¸ **Semantic Scholar API**: No papers found for this query")
                
                return papers
                
            except requests.exceptions.Timeout:
                st.warning(f"âš ï¸ **Semantic Scholar API**: Timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                continue
            
            except Exception as e:
                st.error(f"âŒ **Semantic Scholar API**: Error - {str(e)}")
                break
        
        # Fallback if all fails
        st.warning("âš ï¸ **Semantic Scholar API**: Search failed. Check internet/API key.")
        return []

# ==================== INTELLIGENT MULTI-SOURCE FETCHER ====================
class IntelligentMultiSourceFetcher:
    """Multi-source paper fetcher with intelligent access detection"""
    
    def __init__(self):
        self.fetchers = {
            'arxiv': RealArxivFetcher(),
            'semantic_scholar': SemanticScholarFetcher()
        }
        self.accessor = IntelligentPaperAccessor()
    
    def fetch_papers(self, query: str, sources: List[str], papers_per_source: int) -> List[Dict]:
        """Fetch papers with intelligent access detection"""
        all_papers = []
        source_results = {}
        
        st.write(" ” **Starting multi-source search...**")
        
        # Phase 1: Fetch papers from sources
        for source in sources:
            try:
                if source in self.fetchers:
                    source_display = source.replace('_', ' ').title()
                    
                    with st.spinner(f" “¡ Fetching from {source_display}..."):
                        start_time = time.time()
                        
                        fetcher = self.fetchers[source]
                        papers = fetcher.search_papers(query, papers_per_source)
                        
                        for paper in papers:
                            paper['source'] = source_display
                            paper['fetch_source'] = source
                        
                        all_papers.extend(papers)
                        source_results[source] = len(papers)
                        
                        fetch_time = time.time() - start_time
                        
                        if len(papers) > 0:
                            st.success(f"âœ… **{source_display}**: {len(papers)} papers in {fetch_time:.1f}s")
                        else:
                            st.warning(f"âš ï¸ **{source_display}**: No papers found")
                    
                else:
                    st.error(f"âŒ {source} fetcher not available")
                    source_results[source] = 0
                    
            except Exception as e:
                st.error(f"âŒ **{source.replace('_', ' ').title()}**: {str(e)}")
                source_results[source] = 0
        
        # Phase 2: Intelligent access detection and content extraction
        if all_papers:
            st.write(" §  **Phase 2: Content access detection...**")
            
            processed_papers = []
            accessible_count = 0
            extracted_count = 0
            
            # Progress bar for access detection
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, paper in enumerate(all_papers):
                progress = (i + 1) / len(all_papers)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing paper {i+1}/{len(all_papers)}: {paper.get('title', 'Unknown')[:50]}...")
                
                # Intelligent access check
                enhanced_paper = self.accessor.check_and_extract_paper_content(paper)
                
                if enhanced_paper.get('extracted_content'):
                    extracted_count += 1
                if enhanced_paper.get('pdf_available') or enhanced_paper.get('working_url'):
                    accessible_count += 1
                
                processed_papers.append(enhanced_paper)
                
                # Small delay to show progress
                time.sleep(0.1)
            
            progress_bar.empty()
            status_text.empty()
            
            # Show access detection results
            st.markdown(f"""
            <div class="extraction-status">
            <strong> §  Analysis Complete:</strong><br>
             “„ Total Papers: {len(processed_papers)}<br>
            âœ… Accessible Papers: {accessible_count}<br>
             “œ Content Extracted: {extracted_count}<br>
             ”’ Restricted Access: {len(processed_papers) - accessible_count}
            </div>
            """, unsafe_allow_html=True)
            
            all_papers = processed_papers
        
        # Phase 3: Deduplication and sorting
        unique_papers = self._deduplicate_papers(all_papers)
        unique_papers.sort(key=lambda x: (x.get('year', 0), x.get('month', 0)), reverse=True)
        
        # Show source breakdown
        self._show_source_breakdown(source_results, len(all_papers), len(unique_papers))
        
        return unique_papers
    
    def _show_source_breakdown(self, source_results: Dict, total_before: int, total_after: int):
        """Show detailed source breakdown"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            arxiv_count = source_results.get('arxiv', 0)
            st.markdown(f"""
            <div class="source-status">
            <strong>  ArXiv</strong><br>
            {arxiv_count} papers
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            semantic_count = source_results.get('semantic_scholar', 0)
            st.markdown(f"""
            <div class="source-status">
            <strong> ”¬ Semantic Scholar</strong><br>
            {semantic_count} papers
            </div>
            """, unsafe_allow_html=True)
        
        duplicates_removed = total_before - total_after
        st.info(f"  **Final Summary:** {total_before} papers fetched â†’ {total_after} unique papers (removed {duplicates_removed} duplicates)")
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers with enhanced scoring"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            title_key = re.sub(r'[^\w\s]', '', title)
            title_key = re.sub(r'\s+', ' ', title_key)
            
            if title_key and title_key not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(title_key)
            elif title_key in seen_titles:
                # Enhanced scoring for better paper selection
                existing_idx = None
                for i, existing in enumerate(unique_papers):
                    existing_title = re.sub(r'[^\w\s]', '', existing.get('title', '').lower())
                    existing_title = re.sub(r'\s+', ' ', existing_title)
                    if existing_title == title_key:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    current_score = self._calculate_enhanced_paper_score(paper)
                    existing_score = self._calculate_enhanced_paper_score(unique_papers[existing_idx])
                    
                    if current_score > existing_score:
                        unique_papers[existing_idx] = paper
        
        return unique_papers
    
    def _calculate_enhanced_paper_score(self, paper: Dict) -> int:
        """Calculate enhanced paper quality score"""
        score = 0
        
        # Basic content scoring
        if paper.get('title') and len(paper['title']) > 10:
            score += 2
        if paper.get('abstract') and len(paper['abstract']) > 50:
            score += 3
        if paper.get('authors'):
            score += min(len(paper['authors']), 3)
        
        # Enhanced access scoring
        if paper.get('extracted_content'):
            score += 5  # High score for extracted content
        if paper.get('working_url'):
            score += 3  # Good score for working URLs
        if paper.get('pdf_available'):
            score += 3
        
        # Citation and recency scoring
        citations = int(paper.get('citations') or 0) 
        if citations > 10:
            score += 2
        elif citations > 0:
            score += 1
        
        year = int(paper.get('year') or datetime.now().year)
        if year >= datetime.now().year - 1:
            score += 2
        elif year >= datetime.now().year - 3:
            score += 1
        
        # Source preference (real sources over simulated)
        source = paper.get('fetch_source', '')
        if source in ['arxiv', 'semantic_scholar']:
            score += 3        
        return score

# ==================== CLUSTERING & GAP ANALYSIS (SAME AS BEFORE) ====================
class ImprovedClusterer:
    """Simple area-based clusterer for fast research theme grouping"""
    
    def __init__(self, query: str = ""):
        self.query = query or "general research"  # Store query for relevance biasing

    def cluster_papers(self, papers: List[Dict], query: str = "") -> Dict[int, Dict]:
        effective_query = query or self.query  # Use passed or stored query
        
        if len(papers) < 2:
            return {0: {'name': 'All Papers', 'description': f'Complete research collection relevant to "{effective_query}"', 'papers': papers}}
        
        def identify_research_area(paper: Dict, query: str) -> str:
            abstract = paper.get('abstract', '') or paper.get('title', '')
            text = (abstract + ' ' + query).lower()
            # Keyword mapping for areas (tied to query; expand as needed)
            areas = {
                'machine learning': ['machine learning', 'ml', 'neural', 'deep learning', 'model', 'algorithm'],
                'stock prediction': ['stock', 'prediction', 'financial', 'market', 'forecast', 'trading'],
                'ai applications': ['ai', 'artificial intelligence', 'nlp', 'cv', 'robotics'],
                'other': []  # Default
                }
            for area, keywords in areas.items():
                if any(kw in text for kw in keywords):
                    return area
            return 'general research'  # Fallback

        # Group by research areas, biased toward query
        area_groups = {}
        for paper in papers:
            area = identify_research_area(paper, effective_query)  # Use simple function with query
            if area not in area_groups:
                area_groups[area] = []
            area_groups[area].append(paper)
        
        # Create clusters from groups
        clusters = {}
        for i, (area, papers_list) in enumerate(area_groups.items()):
            # Calculate stats
            avg_citations = sum(int(p.get('citations') or 0) for p in papers_list) / len(papers_list)
            years = [int(p.get('year') or datetime.now().year) for p in papers_list]
            avg_year = sum(years) / len(years) if years else datetime.now().year
            
            # Query-aware description
            description = f"Research papers focusing on {area.lower()} methodologies and applications, relevant to '{effective_query}'"
            
            clusters[i] = {
                'name': area,
                'description': description,
                'paper_count': len(papers_list),
                'avg_citations': round(avg_citations, 1),
                'avg_year': round(avg_year),
                'papers': papers_list
            }
        
        return clusters

class ResearchGapAnalyzer:
    def analyze_gaps(self, papers: List[Dict]) -> Dict[str, List[str]]:
        gaps = {
            'methodological_gaps': [],
            'evaluation_gaps': [], 
            'application_gaps': [],
            'theoretical_gaps': []
        }
        
        # Use extracted content for better gap analysis
        all_content = []
        for paper in papers:
            content = paper.get('extracted_content', '') or paper.get('abstract', '')
            all_content.append(content)
        
        combined_content = ' '.join(all_content).lower()
        
        if 'dataset' in combined_content or 'limited' in combined_content:
            gaps['methodological_gaps'].extend([
                'Limited dataset diversity across different domains and applications',
                'Lack of standardized evaluation protocols for cross-method comparison',
                'Insufficient attention to computational efficiency and scalability issues'
            ])
        
        if 'experiment' in combined_content or 'evaluation' in combined_content:
            gaps['evaluation_gaps'].extend([
                'Need for more comprehensive real-world testing scenarios',
                'Lack of longitudinal studies assessing long-term performance',
                'Limited evaluation on edge cases and adversarial conditions'
            ])
        
        if 'application' in combined_content or 'real-world' in combined_content:
            gaps['application_gaps'].extend([
                'Gap between laboratory results and industrial deployment',
                'Limited integration with existing systems and workflows',
                'Insufficient consideration of regulatory and ethical constraints'
            ])
        
        if 'theoretical' in combined_content or 'analysis' in combined_content:
            gaps['theoretical_gaps'].extend([
                'Lack of theoretical foundations for empirical observations',
                'Limited understanding of failure modes and boundary conditions',
                'Insufficient mathematical analysis of convergence properties'
            ])
        
        return gaps

# ==================== RENDER FUNCTIONS ====================
def render_enhanced_paper_summary(paper: Dict, is_full_text: bool = True):
    """Render enhanced paper summary with content extraction info"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        authors = paper.get('authors', [])
        if authors:
            authors_str = ', '.join(authors[:3])
            if len(authors) > 3:
                authors_str += f' et al. ({len(authors)} total)'
            st.markdown(f"** ‘¥ Authors:** {authors_str}")
        
        source = paper.get('source', 'Unknown')
        year = paper.get('year', 0)
        citations = paper.get('citations', 0)
        st.markdown(f"**  Source:** {source} | **Year:** {year} | **Citations:** {citations}")
        
        # Show content extraction status
        if paper.get('extracted_content'):
            st.markdown("** §  Content:** Extracted and analyzed")
        elif paper.get('enhanced_abstract'):
            st.markdown("** §  Content:** Enhanced abstract available")
    
    with col2:
        if paper.get('extracted_content'):
            st.markdown('<span class="status-extracted"> §  Content Extracted</span>', unsafe_allow_html=True)
        elif is_full_text:
            st.markdown('<span class="status-full">âœ… Full Text Available</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-abstract">âš ï¸ Abstract Only</span>', unsafe_allow_html=True)
    
    # Enhanced AI Summary
    summary = paper.get('ai_summary', {})
    
    if summary:
        st.markdown("---")
        st.markdown("###  ¤– Research Paper Summary")
        
        sections = [
            ('1. Citation / Reference', 'citation'),
            ('2. Problem Statement (What?)', 'problem_statement'),
            ('3. Objective (Why?)', 'objective'), 
            ('4. Methodology (How?)', 'methodology'),
            ('5. Key Findings / Results', 'key_findings'),
            ('6. Limitations / Gaps', 'limitations'),
            ('7. Relevance / Takeaway', 'relevance')
        ]
        
        for title, key in sections:
            content = summary.get(key)
            if content:
                if key == 'key_findings' and isinstance(content, list):
                    findings_html = '<br>'.join([f"â€¢ {finding}" for finding in content])
                    st.markdown(f"""
                    <div class="summary-section">
                    <strong>{title}</strong><br>
                    {findings_html}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="summary-section">
                    <strong>{title}</strong><br>
                    {content}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Enhanced access links
    st.markdown("---")
    
    # Primary access
    working_url = paper.get('working_url') or paper.get('url')
    if working_url:
        access_type = paper.get('access_type', 'direct')
        if access_type == 'direct_pdf':
            st.markdown(f"[**Access Paper (PDF)**]({working_url})")
        else:
            st.markdown(f"[**Access Full Paper**]({working_url})")
    
    # Alternative access links
    alternative_urls = paper.get('alternative_urls', [])
    if alternative_urls:
        st.markdown("**Alternative Access:**")
        for i, alt_url in enumerate(alternative_urls[:3]):  # Show up to 3 alternatives
            st.markdown(f"[Alternative {i+1}]({alt_url})")
    
    # Direct PDF link if different from main URL
    pdf_url = paper.get('pdf_url')  
    if pdf_url and pdf_url != working_url:
        st.markdown(f"[**Direct PDF Download**]({pdf_url})")

def render_suggested_paper(paper: Dict):
    """Render truly restricted paper card"""
    
    st.markdown(f"""
    <div class="suggested-card">
        <h4 style="margin-bottom: 0.5rem; color: #1e293b;">{paper.get('title', 'Unknown Title')}</h4>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.8rem;">
            <strong>Authors:</strong> {', '.join(paper.get('authors', ['Unknown'])[:3])}
            {' et al.' if len(paper.get('authors', [])) > 3 else ''}<br>
            <strong>Source:</strong> {paper.get('source', 'Unknown')} | 
            <strong>Year:</strong> {paper.get('year', 0)} | 
            <strong>Citations:</strong> {paper.get('citations', 0)}
        </p>
        <a href="{paper.get('url', '#')}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500;">
             ”Requires Subscription Access
        </a>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================

# Beautiful Header
st.markdown("""
<div class="main-header">
    <h1>AI Research Assistant</h1>
    <p>Extract, Analyze, and Summarize Research Papers</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### Search")
    
    query = st.text_input(
        "Enter Research Topic",
        placeholder="e.g., machine learning transformers",
        help="Enter research topics for analysis"
    )
    
    st.markdown("### Sources")
    
    use_arxiv = st.checkbox("arXiv", value=True)
    if use_arxiv and not ARXIV_AVAILABLE:
        st.error("ArXiv library missing! Install: `pip install arxiv`")
    
    use_semantic = st.checkbox("Semantic Scholar", value=True) 
    
    st.markdown("### Number of Papers")
    papers_per_source = st.slider(
        "Number of papers per source",
        10, 100, 30,
        help="Select how many papers to fetch per source",
        label_visibility="collapsed"
    )
    
    # Build source list
    sources = []
    if use_arxiv: sources.append('arxiv')
    if use_semantic: sources.append('semantic_scholar')
    
    if sources:
        expected_total = papers_per_source * len(sources)
        st.success(f"Will analyze ~{expected_total} papers")
        source_names = []
        if use_arxiv: 
            source_names.append("ArXiv" + ("" if ARXIV_AVAILABLE else ""))
        if use_semantic: source_names.append("Semantic Scholar")
        
        st.markdown(f"**Sources:** {', '.join(source_names)}")
    else:
        st.error("Please select at least one source!")
    
    # Show content extraction capabilities
    if not BEAUTIFULSOUP_AVAILABLE:
        st.warning("**Install BeautifulSoup for enhanced extraction:** `pip install beautifulsoup4`")
    
    # Start Analysis Button
    if st.button("Start Analysis", type="primary", disabled=st.session_state.processing or not sources or not query):
        if query.strip() and sources:
            st.session_state.processing = True
        
        try:
            start_time = time.time()
            fetcher = IntelligentMultiSourceFetcher()
            papers = fetcher.fetch_papers(query, sources, papers_per_source)
            fetch_time = time.time() - start_time
            
            if len(papers) == 0:
                st.error("No papers found! Try different keywords or sources.")
                st.session_state.processing = False
                st.stop()
            
            original_papers = papers
            # Enhanced relevance filter after fetching (stricter scoring)
            def score_paper_relevance(paper: Dict, query: str) -> float:
                if not query:
                    return 1.0
                query_lower = query.lower()
                query_words = [w for w in query_lower.split() if len(w) > 2]  # Ignore short words like "in", "the"
                if not query_words:
                    return 1.0
                
                title_lower = paper.get('title', '').lower()
                abstract_lower = paper.get('abstract', '').lower()
                content_lower = (paper.get('extracted_content') or '').lower()[:2000]  # Limit content for speed
                
                text_lower = title_lower + ' ' + abstract_lower + ' ' + content_lower
                matches = sum(1 for word in query_words if word in text_lower)
                score = matches / len(query_words) if query_words else 0
                # Bonus for title matches (more relevant)
                title_matches = sum(1 for word in query_words if word in title_lower)
                score += (title_matches / len(query_words)) * 0.5  # Weight title 50%
                return score
            
            # Apply strict filter: Keep only papers with score >= 0.5 (50%+ relevance)
            original_count = len(papers)
            high_relevance_papers = [p for p in papers if score_paper_relevance(p, query) >= 0.5]
            low_relevance_count = original_count - len(high_relevance_papers)

            if low_relevance_count > 0:
                st.info(f"Filtered {low_relevance_count} marginally related papers to focus on high-relevance results.")

            papers = high_relevance_papers
            if len(papers) == 0:
                st.warning("No highly relevant papers found. Showing all fetched results.")
                papers = original_papers  # Fallback to all if empty (add original_papers = papers before filter if needed)
            
            st.success(f"**Total: {len(papers)} highly relevant papers** analyzed in {fetch_time:.1f}s (filtered from {original_count})")
            
            # ... (rest of the try-block continues: summaries spinner, clustering, etc.)

           
            with st.spinner("Generating enhanced AI summaries..."):
                starttime = time.time()
                summarizer = FullPaperSummarizer(model="gpt-4o-mini")  # Use your OpenAI model; set OPENAI_API_KEY env var
                fulltextpapers = []
                suggestedpapers = []
                for paper in papers:
                    isfulltext = bool(paper.get('pdf_available') or paper.get('full_text') or paper.get('extracted_content'))
                    # Generate AI summary using LLM with query and full-text check
                    ai_summary = summarizer.summarize_paper(
                        paper, 
                        use_full_text=isfulltext, 
                        timeout=120, 
                        query=query  # Pass query for relevance
                    )
                    if ai_summary and isinstance(ai_summary, dict):
                        paper['ai_summary'] = ai_summary
                        if isfulltext:
                            fulltextpapers.append(paper)
                        else:
                            suggestedpapers.append(paper)
                    else:
                        # Fallback to conservative summary if LLM fails/unavailable
                        meta = summarizer._prepare_meta(paper)  # Private but accessible; prepare metadata
                        conservative = summarizer.conservative_summary(meta, query=query)
                        paper['ai_summary'] = conservative
                        if isfulltext:
                            fulltextpapers.append(paper)
                        else:
                            suggestedpapers.append(paper)
                summarytime = time.time() - starttime
                st.success(f"Generated {len(papers)} enhanced summaries in {summarytime:.1f}s")

            with st.spinner("Analyzing research themes..."):
                start_time = time.time()
                clusterer = ImprovedClusterer(query=query)  # Pass query for relevance
                clusters = clusterer.cluster_papers(papers, query=query)
                cluster_time = time.time() - start_time
                st.success(f"Identified {len(clusters)} research themes in {cluster_time:.1f}s")
            
            # Store results
            st.session_state.papers_data = papers
            st.session_state.full_text_papers = fulltextpapers
            st.session_state.suggested_papers = suggestedpapers
            st.session_state.clusters = clusters
            st.session_state.processing = False
                
            st.balloons()
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.session_state.processing = False

    # Clear Results Button
    if st.button("Clear Results", type="secondary"):
        st.session_state.papers_data = []
        st.session_state.full_text_papers = []
        st.session_state.suggested_papers = []
        st.session_state.clusters = {}
        st.rerun()
    
    # Clean footer
    st.markdown("---")
    st.markdown("*Intelligent research with content extraction*")

# ==================== MAIN CONTENT ====================

if st.session_state.papers_data:
    # Enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(st.session_state.papers_data)}</div>
            <div class="metric-label">Papers Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accessible_count = len(st.session_state.full_text_papers)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{accessible_count}</div>
            <div class="metric-label">Accessible Papers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        extracted_count = len([p for p in st.session_state.papers_data if p.get('extracted_content')])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{extracted_count}</div>
            <div class="metric-label">Content Extracted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sources_count = len(set([p.get('source', 'unknown') for p in st.session_state.papers_data]))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{sources_count}</div>
            <div class="metric-label">Sources Used</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Clean tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Dashboard", 
        " Papers & Summaries", 
        " Research Gaps",
        " Restricted Reading"
    ])
    
    with tab1:
        st.markdown("### Research Dashboard")
        st.markdown("*Intelligent analysis with enhanced content extraction*")
        
        if st.session_state.clusters:
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_names = []
                cluster_counts = []
                for cluster_id, cluster_info in st.session_state.clusters.items():
                    cluster_names.append(cluster_info['name'])
                    cluster_counts.append(cluster_info['paper_count'])
                
                colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#667eea', '#764ba2']
                
                fig = px.pie(
                    values=cluster_counts,
                    names=cluster_names,
                    title="Research Areas Distribution",
                    color_discrete_sequence=colors
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    font_family="Inter",
                    title_font_size=16,
                    font_size=12,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                citation_data = []
                area_names = []
                for cluster_info in st.session_state.clusters.values():
                    citation_data.append(cluster_info.get('avg_citations', 0))
                    area_names.append(cluster_info['name'])
                
                fig = px.bar(
                    x=citation_data,
                    y=area_names,
                    orientation='h',
                    title="Average Citations by Research Area",
                    color=citation_data,
                    color_continuous_scale=["#f8fafc", "#667eea", "#764ba2"]
                )
                fig.update_layout(
                    font_family="Inter",
                    title_font_size=16,
                    font_size=12,
                    xaxis_title="Average Citations",
                    yaxis_title="Research Area"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced cluster cards
            st.markdown("### Research Themes")
            
            for cluster_id, cluster_info in st.session_state.clusters.items():
                # Filter papers to those relevant to query (simple keyword match for safety)
                relevant_papers = [p for p in cluster_info['papers'] if any(word in (p.get('title', '') + p.get('abstract', '')).lower() for word in query.lower().split())]
                extracted_in_cluster = len([p for p in relevant_papers if p.get('extracted_content')])  # Update count
                cluster_info['papers'] = relevant_papers  # Override for display
                
                st.markdown(f"""
                <div class="cluster-card">
                    <div class="cluster-title">{cluster_info['name']}</div>
                    <p style="color: #64748b; margin-bottom: 1rem;">{cluster_info['description']}</p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                             “{cluster_info['paper_count']} papers
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                             {extracted_in_cluster} content extracted
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                              {cluster_info.get('avg_citations', 0)} avg citations
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                             “… ~{cluster_info.get('avg_year', 2024)}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Complete analysis to see research themes and dashboard metrics.")
    
    with tab2:
        st.markdown("### Papers & Summaries")
        st.markdown("*Enhanced summaries from accessible and extracted content (newest first)*")
        
        if not st.session_state.full_text_papers:
            st.info("No accessible papers found. Check sources or try different keywords.")
        else:
            # Show ALL accessible papers
            total_papers = len(st.session_state.full_text_papers)
            extracted_papers = len([p for p in st.session_state.full_text_papers if p.get('extracted_content')])
            
            st.markdown(f"**Showing all {total_papers} accessible papers ({extracted_papers} with extracted content)**")
            
            papers_per_page = 20
            total_pages = (len(st.session_state.full_text_papers) - 1) // papers_per_page + 1
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1)) - 1
                start_idx = page * papers_per_page
                end_idx = start_idx + papers_per_page
                papers_to_show = st.session_state.full_text_papers[start_idx:end_idx]
                st.markdown(f"*Showing papers {start_idx + 1}-{min(end_idx, len(st.session_state.full_text_papers))} of {len(st.session_state.full_text_papers)}*")
            else:
                papers_to_show = st.session_state.full_text_papers
            
            for i, paper in enumerate(papers_to_show, 1):
                content_indicator = ""
                if paper.get('extracted_content'):
                    content_indicator = "  § "
                elif paper.get('working_url'):
                    content_indicator = " âœ…"
                
                with st.expander(f"{i + (page * papers_per_page if 'page' in locals() else 0)}. {paper.get('title', 'Unknown Title')}{content_indicator} ({paper.get('year', 0)})"):
                    render_enhanced_paper_summary(paper, is_full_text=True)
    
    with tab3:
        st.markdown("### Research Gaps Analysis")
        st.markdown("*Enhanced gap analysis using extracted content*")
        
        if st.session_state.papers_data:
            gap_analyzer = ResearchGapAnalyzer()
            gaps = gap_analyzer.analyze_gaps(st.session_state.papers_data)
            
            for gap_type, gap_list in gaps.items():
                if gap_list:
                    gap_title = gap_type.replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="gap-card">
                        <h4 style="margin-bottom: 0.8rem; color: #7c3aed;">{gap_title}</h4>
                        <ul style="color: #374151; margin: 0; padding-left: 1.5rem;">
                    """, unsafe_allow_html=True)
                    
                    for gap in gap_list:
                        st.markdown(f"<li style='margin-bottom: 0.5rem;'>{gap}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.info("Complete paper analysis to identify research gaps and opportunities.")
    
    with tab4:
        st.markdown("### Restricted Reading")
        st.markdown("*Papers requiring institutional or subscription access*")
        
        st.markdown("""
        <div class="warning-box">
         ”<strong>Restricted Access:</strong> Access to these papers requires a subscription or paid access.
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.suggested_papers:
            st.success("Excellent! All papers are accessible. Check 'Papers & Summaries' for complete analysis with extracted content.")
        else:
            st.markdown(f"**{len(st.session_state.suggested_papers)} papers requiring paid/institutional access**")
            
            for i, paper in enumerate(st.session_state.suggested_papers, 1):
                render_suggested_paper(paper)

else:
    # Enhanced welcome screen
    st.markdown("### Intelligent Research Analysis")
    
    steps = [
        {
            'title': 'Enter Research Topic',
            'description': 'Type your research keywords for intelligent multi-source analysis',
            'expected': 'Focused topics yield better content extraction results'
        },
        {
            'title': 'Select Enhanced Sources', 
            'description': 'Choose from real APIs with intelligent access detection capabilities',
            'expected': f'ArXiv: {"" if ARXIV_AVAILABLE else ""}, Semantic Scholar:'
        },
        {
            'title': 'Set Paper Count',
            'description': 'Choose papers per source (10-100) - system will intelligently analyze accessibility',
            'expected': 'More papers = comprehensive analysis but longer processing time'
        },
        {
            'title': 'Start Intelligent Analysis',
            'description': 'AI system will fetch, extract content, detect access, and generate enhanced summaries',
            'expected': '30-90 seconds for complete intelligent analysis with content extraction'
        }
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="welcome-step">
            <div style="display: flex; align-items: flex-start;">
                <div class="step-number">{i}</div>
                <div>
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">{step['title']}</h4>
                    <p style="color: #64748b; margin: 0 0 0.8rem 0; line-height: 1.5;">{step['description']}</p>
                    <div style="background: #f1f5f9; padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.9rem; color: #475569;">
                        <strong>Expected:</strong> {step['expected']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced capabilities
    st.markdown("### Enhanced Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ** Intelligent Content Extraction**
        - Detects accessible papers automatically
        - Extracts full text content when available  
        - Enhanced summaries from extracted content
        - Direct repository link generation
        
        **  Smart Classification**
        - Only truly restricted papers in "Suggested Reading"
        - Accessible papers get full analysis
        - Content extraction status indicators
        - Enhanced citation and relevance analysis
        """)
    
    with col2:
        st.markdown("""
        ** Advanced Access Detection**
        - Multiple repository URL generation
        - Working link identification
        - Alternative access point discovery
        - PDF availability verification
        
        ** Enhanced Analysis**
        - Content-aware gap analysis
        - Extracted content-based clustering
        - Smart paper scoring and deduplication
        - Real-time access status updates
        """)
    
    # Dependencies and capabilities
    st.markdown("### System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ARXIV_AVAILABLE:
            st.success(" **ArXiv API** - Real papers available")
        else:
            st.error("**ArXiv** - Install: `pip install arxiv`")
    
    with col2:
        st.success("**Semantic Scholar** - Enhanced API ready")
    
    with col3:
        if BEAUTIFULSOUP_AVAILABLE:
            st.success("**Content Extraction** - Advanced parsing")
        else:
            st.warning("**Install BeautifulSoup** - `pip install beautifulsoup4`")
    
    st.markdown("### Example Results")
    st.markdown("**Query:** `deep learning transformers` â†’ **Expected:** 60-300 papers â†’ **Intelligent Analysis:** Content extraction, access detection, enhanced summaries â†’ **Time:** 30-90 seconds")

# Clean footer
if not st.session_state.processing:
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'> §  Intelligent research assistant with content extraction</div>", unsafe_allow_html=True)
