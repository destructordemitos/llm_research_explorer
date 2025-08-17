import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, Any
import google.generativeai as genai
from fpdf import FPDF
import base64
import tempfile
import os
import uuid
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# --- Performance Optimization: Caching Data Loading ---
@st.cache_data
def load_ror_data():
    """Loads and caches the ROR dataset."""
    try:
        return pd.read_csv("ror-data.csv")
    except FileNotFoundError:
        st.error("ROR data file not found. Please ensure 'ror-data.csv' is in the app directory.")
        return pd.DataFrame(columns=["name", "id", "country.country_name", "country.country_code"])

@st.cache_data
def load_topic_options():
    """Loads and caches the OpenAlex topics."""
    try:
        topics_df = pd.read_csv("openalex-topics.csv")
        return sorted(
            set(topics_df["topic_name"].dropna().tolist() +
                topics_df["subfield_name"].dropna().tolist() +
                topics_df["field_name"].dropna().tolist())
        )
    except FileNotFoundError:
        st.error("Topics data file not found. Please ensure 'openalex-topics.csv' is in the app directory.")
        return []

# --- API and Data Processing Helpers ---
def reconstruct_abstract(abstract_inverted_index: Dict[str, List[int]]) -> str:
    """Reconstructs an abstract from OpenAlex's inverted index format."""
    if not abstract_inverted_index:
        return "No abstract available."
    
    words_map = {pos: word for word, positions in abstract_inverted_index.items() for pos in positions}
    return " ".join(words_map[i] for i in sorted(words_map))

def format_author_affiliations(authorships: List[Dict]) -> str:
    """
    Formats author and their primary institution affiliations into a readable string.
    Example: 'John Doe (University of A), Jane Smith (Research Corp B)'
    """
    if not authorships:
        return "N/A"

    affiliation_parts = []
    for authorship in authorships:
        author_name = authorship.get('author', {}).get('display_name', 'N/A')
        
        # Get the first institution for this author, if available
        institutions = authorship.get('institutions', [])
        institution_name = institutions[0].get('display_name', 'N/A') if institutions else 'N/A'
        
        affiliation_parts.append(f"{author_name} ({institution_name})")
        
    return ", ".join(affiliation_parts)

def extract_authors_and_institutions(authorships: List[Dict]) -> tuple[str, str]:
    """Extracts comma-separated author and institution names from authorships."""
    if not authorships:
        return "N/A", "N/A"
    
    authors_list = [authorship.get('author', {}).get('display_name', 'N/A') for authorship in authorships]
    institutions_list = sorted(list(set(
        inst.get('display_name', 'N/A') 
        for authorship in authorships 
        for inst in authorship.get('institutions', [])
    )))
    
    return ", ".join(authors_list), ", ".join(institutions_list)

def process_openalex_records(records: List[Dict], institution_name: str = None) -> List[Dict]:
    """Processes a list of records from OpenAlex by adding formatted author affiliations and abstracts."""
    for record in records:
        authorships = record.get("authorships", [])
        record["author_affiliations"] = format_author_affiliations(authorships)
        record["abstract"] = reconstruct_abstract(record.get("abstract_inverted_index"))
        
        # Determine the primary institution for the record if not already provided
        if institution_name:
            record["institution_name"] = institution_name
        elif authorships and authorships[0].get('institutions'):
            record["institution_name"] = authorships[0]['institutions'][0].get('display_name', 'N/A')
        else:
            record["institution_name"] = "N/A"
    return records

def search_openalex_task(query: str, start_year: int, ror_id: str = None, country_code: str = None, identifier: str = "global") -> Dict:
    """A single, targeted search task for OpenAlex API, designed for parallel execution."""
    base_url = "https://api.openalex.org/works"
    max_results = 100
    filters = [f"title_and_abstract.search:{query}", f"from_publication_date:{start_year}-01-01"]
    if ror_id:
        filters.append(f"institutions.ror:{ror_id}")
    if country_code:
        filters.append(f"institutions.country_code:{country_code}")

    params = {
        "filter": ",".join(filters),
        "sort": "cited_by_count:desc",
        "per_page": max_results,
        "page": 1,
        "mailto": st.secrets.get("openalex_email", "user@example.com"),
        "select": "id,title,publication_year,abstract_inverted_index,authorships,fwci,cited_by_count"
    }
    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        results = response.json().get("results", [])
        return {"identifier": identifier, "records": results, "error": None}
    except requests.RequestException as e:
        return {"identifier": identifier, "records": [], "error": str(e)}

def fetch_all_data_in_parallel(topic: str, start_year: int, institutions: List[Dict], countries: List[Dict]) -> Dict:
    """
    Fetches all required data from OpenAlex in parallel to optimize performance.
    """
    tasks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit institution tasks
        for inst in institutions:
            tasks.append(executor.submit(search_openalex_task, topic, start_year, ror_id=inst['id'], identifier=inst['name']))
        
        # Submit country tasks
        for country in countries:
            tasks.append(executor.submit(search_openalex_task, topic, start_year, country_code=country['code'], identifier=country['name']))

        # Submit global task
        tasks.append(executor.submit(search_openalex_task, topic, start_year, identifier="global"))

        results = {"institutions": {}, "countries": {}, "global": [], "errors": []}
        for future in as_completed(tasks):
            try:
                result = future.result()
                identifier = result["identifier"]
                if result["error"]:
                    results["errors"].append(f"Failed to fetch data for {identifier}: {result['error']}")
                    continue
                
                processed_records = process_openalex_records(result["records"])

                if identifier in [inst['name'] for inst in institutions]:
                    results["institutions"][identifier] = processed_records
                elif identifier in [country['name'] for country in countries]:
                    results["countries"][identifier] = processed_records
                elif identifier == "global":
                    results["global"] = processed_records
            except Exception as e:
                results["errors"].append(f"A task failed with exception: {e}")
    return results

def format_works(title: str, works: List[Dict]) -> str:
    """Formats a list of works into a string for the LLM context, using linked author affiliations."""
    output_parts = [f"\n\n### {title}\n"]
    for w in works:
        parts = [
            f"Id: {w.get('id', 'N/A')}",
            f"Title: {w.get('title', 'No Title')}",
            f"Year: {w.get('publication_year', 'N/A')}",
            f"Citations: {w.get('cited_by_count', 0)}",
            f"Authors: {w.get('author_affiliations', 'N/A')}",
        ]
        output_parts.append(" | ".join(parts))
    return "\n".join(output_parts)

# --- PDF and LLM Helpers (with rate limit handling) ---

class PDF(FPDF):
    """Custom PDF class to create styled headers and footers."""
    def header(self):
        # Set up a logo if you have one, e.g., self.image('logo.png', 10, 8, 33)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AltNex Research Intelligence Summary', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 8, f'Generated on: {time.strftime("%Y-%m-%d")}', 0, 1, 'C')
        self.ln(10) # Add a break after the header

    def footer(self):
        self.set_y(-15) # Position 1.5 cm from bottom
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def generate_pdf(summary_text: str) -> str:
    """Generates a styled, downloadable PDF link from text, with secure file handling."""
    try:
        pdf = PDF(orientation='P', unit='mm', format='A4')
        pdf.alias_nb_pages() # Enable page numbering
        
        # --- KEY FIX: Add a font that supports Unicode characters like '•' ---
        # The 'DejaVu' font family is a good choice for broad Unicode support.
        # This tells FPDF to use UTF-8 encoding.
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Process the summary text line by line for basic Markdown styling
        for line in summary_text.split('\n'):
            line = line.strip()
            if not line:
                pdf.ln(4) # Add space for empty lines
                continue

            if line.startswith('### '):
                pdf.set_font('DejaVu', 'B', 14)
                pdf.multi_cell(0, 8, line.replace('### ', '').strip())
                pdf.ln(2)
            elif line.startswith('## '):
                pdf.set_font('DejaVu', 'B', 16)
                pdf.multi_cell(0, 10, line.replace('## ', '').strip())
                pdf.ln(3)
            elif line.startswith('**') and line.endswith('**'):
                pdf.set_font('DejaVu', 'B', 12)
                pdf.multi_cell(0, 8, line.strip('*'))
            elif line.startswith('* '):
                pdf.set_font('DejaVu', '', 12)
                # Now we can safely use the bullet character
                pdf.multi_cell(0, 8, f"  •  {line.strip('* ')}")
            else:
                pdf.set_font('DejaVu', '', 12)
                # The text is now handled correctly by the Unicode font, no manual encoding needed.
                pdf.multi_cell(0, 8, line)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            tmpfile.seek(0)
            pdf_data = tmpfile.read()
        
        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        return f'<a href="data:application/pdf;base64,{b64_pdf}" download="summary_analysis.pdf" style="background: var(--col-accent); color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; display: inline-block;">📄 Download Styled PDF</a>'
    except Exception as e:
        # Provide a more helpful error message if the font is missing
        if "FPDF error: Can't open file" in str(e):
            st.error("Failed to generate PDF: The 'DejaVu' font file is missing. Please ensure 'DejaVuSans.ttf' and 'DejaVuSans-Bold.ttf' are in your project directory.")
        else:
            st.error(f"Failed to generate PDF: {e}")
        return ""

def exponential_backoff_retry(func, max_retries=3, base_delay=3):
    """Retry function with exponential backoff for rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise e

def safe_generate_content(model, prompt, max_retries=3):
    """Safely generate content with rate limit handling."""
    return exponential_backoff_retry(lambda: model.generate_content(prompt), max_retries)

# --- UI Rendering Functions ---
def render_data_summary(institution_counts, country_counts, global_count, topic, start_year):
    """Renders the data summary cards with counts."""
    st.markdown("""
    <style>
    .ds-wrap{background:var(--col-panel);border-radius:var(--radius-md);padding:16px;margin:8px 0;}
    .ds-header{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;justify-content:center;}
    .ds-tag{background:var(--col-accent);color:#fff;border-radius:20px;padding:6px 14px;font-size:13px;font-weight:500;}
    .ds-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    .ds-card{background:rgba(255,255,255,0.05);border-radius:10px;padding:16px;border-left:4px solid var(--col-accent);}
    .ds-card-title{margin:0 0 12px 0;font-weight:600;font-size:16px;color:var(--col-accent);}
    .ds-list{margin:0;padding:0;list-style:none;}
    .ds-list li{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);}
    .ds-list li:last-child{border-bottom:none;}
    .ds-inst-name{flex:1;font-size:14px;line-height:1.3;}
    .ds-count{background:var(--col-chat);color:#fff;border-radius:12px;padding:2px 8px;font-size:12px;font-weight:500;min-width:40px;text-align:center;}
    @media (max-width: 768px) {.ds-grid{grid-template-columns:1fr;}}
    </style>
    """, unsafe_allow_html=True)

    inst_items = "".join([f"<li><span class='ds-inst-name'>{name}</span><span class='ds-count'>{count}</span></li>" for name, count in institution_counts.items()])
    country_items = "".join([f"<li><span class='ds-inst-name'>{name}</span><span class='ds-count'>{count}</span></li>" for name, count in country_counts.items()])
    
    st.markdown(f"""
    <div class="ds-wrap">
      <div class="ds-header">
        <span class='ds-tag'>📚 {topic}</span>
        <span class='ds-tag'>📅 From {start_year}</span>
      </div>
      <div class="ds-grid">
        <div class="ds-card">
          <h3 class="ds-card-title">🏛️ Institutions</h3>
          <ul class='ds-list'>{inst_items}</ul>
        </div>
        <div class="ds-card">
          <h3 class="ds-card-title">🌍 Countries</h3>
          <ul class='ds-list'>{country_items}</ul>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main App ---
# Load initial data
ror_df = load_ror_data()
topic_options = load_topic_options()

# Initialize GenAI Model
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
except Exception as e:
    st.error(f"Error initializing the Generative AI model: {e}")
    model = None

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ran_query" not in st.session_state:
    st.session_state.ran_query = False

# --- UI Layout ---
st.image("images/altusnexus_logo.png", width=200)
st.title("AltNex Research Intelligence")
st.write("Query research data from OpenAlex and interact with it using an LLM. By default, the app fetches the top 100 most cited works for your selections.")

# --- Filters Section ---
with st.expander("🔍 Filters & Search Configuration", expanded=not st.session_state.ran_query):
    topic = st.selectbox("Topic", options=[""] + topic_options, format_func=lambda x: "Select a topic..." if x == "" else x)
    start_year = st.number_input("Start Year", min_value=1980, max_value=2025, value=2020, step=1)
    
    countries = sorted(ror_df["country.country_name"].dropna().unique().tolist())
    selected_countries_names = st.multiselect("Select Countries", options=countries, default=[])
    
    if selected_countries_names:
        filtered_ror_df = ror_df[ror_df["country.country_name"].isin(selected_countries_names)]
    else:
        filtered_ror_df = ror_df
    
    available_institutions = sorted(filtered_ror_df["name"].dropna().unique().tolist())
    selected_institution_names = st.multiselect("Select Institutions (up to 5)", options=available_institutions, default=[])

    if len(selected_institution_names) > 5:
        st.warning("Please select a maximum of 5 institutions for optimal performance.")
        selected_institution_names = selected_institution_names[:5]

    run_query = st.button("🚀 Run Analysis", use_container_width=True)

# --- Main Logic ---
if run_query:
    if not all([topic, start_year, selected_institution_names]):
        st.error("Please select a topic, start year, and at least one institution.")
    else:
        st.session_state.ran_query = True
        st.session_state.summary_generated = False # Reset summary generation flag
        st.session_state.chat_history = [] # Clear previous chat

        with st.spinner("Fetching and processing data in parallel... This may take a moment."):
            # Prepare institution and country details for parallel fetching
            institutions_to_fetch = ror_df[ror_df['name'].isin(selected_institution_names)][['name', 'id']].to_dict('records')
            countries_to_fetch = ror_df[ror_df['country.country_name'].isin(selected_countries_names)][['country.country_name', 'country.country_code']].rename(columns={'country.country_name': 'name', 'country.country_code': 'code'}).drop_duplicates().to_dict('records')

            # Fetch all data
            fetched_data = fetch_all_data_in_parallel(topic, start_year, institutions_to_fetch, countries_to_fetch)
            
            if fetched_data["errors"]:
                for error in fetched_data["errors"]:
                    st.warning(error)

            # --- Consolidate and Store Results in Session State ---
            all_institution_records = [record for inst_records in fetched_data["institutions"].values() for record in inst_records]
            
            # Build context for LLM
            context = format_works("Selected Institution Results", all_institution_records)
            for country_name, records in fetched_data["countries"].items():
                context += format_works(f"Country Context: {country_name}", records)
            context += format_works("Global Context", fetched_data["global"])
            st.session_state.context = context

            # Store data for UI display
            st.session_state.institution_counts = {name: len(records) for name, records in fetched_data["institutions"].items()}
            st.session_state.country_counts = {name: len(records) for name, records in fetched_data["countries"].items()}
            st.session_state.global_count = len(fetched_data["global"])
            st.session_state.topic = topic
            st.session_state.start_year = start_year
        
        st.rerun()

# --- Display Results and AI Analysis (if query has been run) ---
if st.session_state.ran_query:
    # Display Data Summary
    render_data_summary(
        st.session_state.institution_counts,
        st.session_state.country_counts,
        st.session_state.global_count,
        st.session_state.topic,
        st.session_state.start_year
    )

    # Generate initial AI summary only once per query
    if not st.session_state.get('summary_generated', False) and model:
        with st.spinner("🤖 Generating initial analysis..."):
            summary_prompt = f"""
            You are a research strategy consultant. Based on the provided research data, provide a concise summary analysis. 
            Highlight the main strengths and weaknesses of the selected institutions compared to their country and the global landscape for the topic '{st.session_state.topic}'.
            Keep it brief and focused on actionable insights.

            Data (truncated for summary):
            {st.session_state.context[:15000]}
            """
            try:
                summary_response = safe_generate_content(model, summary_prompt)
                st.session_state.ai_summary = summary_response.text
                st.session_state.summary_generated = True
            except Exception as e:
                st.session_state.ai_summary = f"Could not generate summary due to an error: {e}"

    if 'ai_summary' in st.session_state:
        with st.expander("🤖 AI Generated Summary Analysis", expanded=True):
            st.markdown(st.session_state.ai_summary)
            st.markdown(generate_pdf(st.session_state.ai_summary), unsafe_allow_html=True)

    # --- LLM Chat Interface ---
    # The LLM will now always use the maximum available context for the best possible answers.
    
    # Display chat history from session state
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_prompt := st.chat_input("Ask a follow-up question about the research data..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Get LLM response
        with st.spinner("Thinking..."):
            # Build prompt with dynamic context size
            chat_history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in st.session_state.chat_history[-5:-1]]) # Pass recent history
            
            # Always use the maximum context for the most detailed answers.
            context_limit = 25000
            truncated_context = st.session_state.context[:context_limit]

            full_prompt = f"""
            You are an experienced and detailed-focused research strategy and collaboration consultant. Answer the user's question based on the provided research data and conversation history.
            When you mention a specific paper, provide a link to its OpenAlex page using this format: `https://openalex.org/W...` where `W...` is the paper's ID.
            Be concise but thorough in your analysis, and provide actionable insights where possible, with paricular focus on internatIonal collaboration and research collaboration opportunities.

            CONVERSATION HISTORY:
            {chat_history_str}

            RESEARCH DATA:
            {truncated_context}

            USER'S QUESTION:
            {user_prompt}

            YOUR ANSWER:
            """
            try:
                response = safe_generate_content(model, full_prompt)
                response_text = response.text
            except Exception as e:
                response_text = f"Sorry, I could not generate a response. Error: {e}"
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        
        # Set a flag to trigger auto-scrolling on the next run
        st.session_state.scroll_to_bottom = True
        st.rerun()

    # Auto-scroll to the bottom if a new message was sent
    if st.session_state.get("scroll_to_bottom", False):
        st.components.v1.html(
            """
            <script>
                const main = window.parent.document.querySelector('section.main');
                if (main) {
                    main.scrollTo(0, main.scrollHeight);
                }
            </script>
            """,
            height=0
        )
        st.session_state.scroll_to_bottom = False # Reset the flag