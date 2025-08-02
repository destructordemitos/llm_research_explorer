import streamlit as st
import pandas as pd
import requests
from typing import List
import google.generativeai as genai
from fpdf import FPDF
import base64

# --- Load ROR Data ---
try:
    ror_df = pd.read_csv("ror-data.csv")
except FileNotFoundError:
    st.error("ROR data file not found. Please ensure 'ror-data.csv' is in the app directory.")
    ror_df = pd.DataFrame(columns=["name", "id", "country.country_code"])  # Create an empty DataFrame as fallback

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #2c1e4a;
        color: white;
    }
    .stApp {
        background-color: #2c1e4a;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: white !important;
    }
    div.stButton > button {
        width: 100%;
        background-color: #d73cbe;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5em 1em;
        font-size: 16px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #b82fa1;
    }
    .filter-section {
        background-color: #3b2a5c; /* 20% lighter than #2c1e4a */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .results-section {
        background-color: #3b2a5c;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Add Logo ---
st.image("images/altusnexus_logo.png", width=200)

# --- Streamlit Layout ---
st.title("Altus Nexus Research Intelligence")
st.write(
    "This app allows you to query research data in granular topics from OpenAlex and interact with it using a large language model (LLM). "
    "By default, the model will select the top 100 works based on citation count, but you can adjust the filters to explore different aspects of the research landscape."
)

# --- Filters Section ---
st.subheader("Filters")

# First row: Topic and Start Year
col1, col2 = st.columns([7, 3])  # Adjusted column widths: 70% for col1, 30% for col2
with col1:
    topic = st.text_input("Topic", key="topic_input", placeholder="e.g., microbiology")
with col2:
    start_year = st.number_input("Start Year", min_value=1980, max_value=2025, value=2020, step=1, key="year_input")

# Second row: Institution Name
institution_name = st.selectbox(
    "Institution",
    options=[""] + sorted(ror_df["name"].dropna().unique().tolist()),
    placeholder="Type and select an institution",
    format_func=lambda x: "Type and select" if x == "" else x,
    key="institution_input"
)

# Run Query Button
run_query = st.button("Run Query", use_container_width=True)


# Placeholders for dynamic content
results_placeholder = st.empty()
chat_placeholder = st.empty()

# --- Initialize GenAI Model ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
except Exception as e:
    st.error(f"Error initializing the Generative AI model: {e}")
    model = None

# --- Helper Functions ---
def search_openalex(query, institution_ids=None, country_code=None, per_page=50, sort_by='cited_by_count', start_year=None):
    base_url = "https://api.openalex.org/works"
    select_fields = (
        "id,title,publication_year,open_access,abstract_inverted_index,"
        "authorships,fwci,cited_by_count,cited_by_percentile_year,"
        "countries_distinct_count,institutions_distinct_count,"
        "keywords,topics,concepts"
    )

    filters = []
    if query:
        filters.append(f"title_and_abstract.search:{query}")
    if institution_ids:
        filters.append(f"institutions.ror:{'|'.join(institution_ids)}")
    if country_code:
        filters.append(f"institutions.country_code:{country_code}")
    if start_year:
        filters.append(f"from_publication_date:{start_year}-01-01")

    params = {
        "sort": f"{sort_by}:desc",
        "per_page": per_page,
        "mailto": "destructordemitos@gmail.com",
        "filter": ",".join(filters),
        "select": select_fields
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch from OpenAlex: {e}")
        return []

def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return "No abstract available."
    
    words_map = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            words_map[pos] = word
            
    # Sort by position index and join the words
    return " ".join(words_map[i] for i in sorted(words_map))

def extract_authors_and_institutions(authorships):
    authors_list = []
    institutions_list = []
    if not authorships:
        return "N/A", "N/A"
        
    for authorship in authorships:
        author_name = authorship.get('author', {}).get('display_name', 'N/A')
        authors_list.append(author_name)
        
        for inst in authorship.get('institutions', []):
            inst_name = inst.get('display_name', 'N/A')
            if inst_name not in institutions_list:
                institutions_list.append(inst_name)
                
    return ", ".join(authors_list) if authors_list else "N/A", ", ".join(institutions_list) if institutions_list else "N/A"

def format_works(title, works):
    formatted_output = f"\n\n### {title}\n"
    for w in works:
        output_parts = [
            f"- Title: {w.get('title', 'No Title')}",
            f"Publication Year: {w.get('publication_year', 'N/A')}",
            f"Citations: {w.get('cited_by_count', 0)}",
            f"Authors: {w.get('authors_list', 'N/A')}",
            f"Institutions: {w.get('institutions_list', 'N/A')}",
            f"Abstract: {w.get('abstract', 'N/A')[:200]}..."
        ]
        
        fwci_value = w.get('fwci')
        if fwci_value is not None:
            output_parts.append(f"FWCI: {fwci_value:.2f}" if isinstance(fwci_value, (int, float)) else f"FWCI: {fwci_value}")

        # ... (Add other fields as needed) ...
        
        formatted_output += ", ".join(output_parts) + "\n"
    return formatted_output


# --- Streamlit UI ---
# --- Main Logic ---
if run_query:
    if not all([institution_name, topic, start_year]):
        st.error("Please fill in all filter fields before running the query.")
    else:
        inst_row = ror_df[ror_df['name'] == institution_name]
        if inst_row.empty:
            st.error("Lookup failed: The selected institution name was not found in the CSV.")
        else:
            institution_id = inst_row.iloc[0]['id']
            country_code = inst_row.iloc[0].get('country.country_code', '')

            with st.spinner("Fetching data from OpenAlex..."):
                inst_works = search_openalex(topic, institution_ids=[institution_id], start_year=start_year)
                country_works = search_openalex(topic, country_code=country_code, start_year=start_year)
                global_works = search_openalex(topic, start_year=start_year)

            # --- Process Data and Display ---
            all_works = [inst_works, country_works, global_works]
            for work_list in all_works:
                for work in work_list:
                    work['abstract'] = reconstruct_abstract(work.get('abstract_inverted_index'))
                    work['authors_list'], work['institutions_list'] = extract_authors_and_institutions(work.get('authorships'))

            cols_to_display = ["title", "publication_year", "authors_list", "institutions_list", "cited_by_count", "fwci", "abstract"]

            # --- Results Section ---
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            st.subheader("Results")
            with results_placeholder.container():
                st.subheader("Institution Results")
                inst_df = pd.DataFrame(inst_works)
                st.dataframe(inst_df[[col for col in cols_to_display if col in inst_df.columns]])

                st.subheader("Country-Level Results")
                country_df = pd.DataFrame(country_works)
                st.dataframe(country_df[[col for col in cols_to_display if col in country_df.columns]])

                st.subheader("Global Results")
                global_df = pd.DataFrame(global_works)
                st.dataframe(global_df[[col for col in cols_to_display if col in global_df.columns]])
            st.markdown("</div>", unsafe_allow_html=True)

            # --- Prepare Context for LLM ---
            context = (
                format_works(f"Top 100 Works - {institution_name}", inst_works[:100]) +
                format_works(f"Top 100 Works - Country ({country_code})", country_works[:100]) +
                format_works("Top 100 Works - Global", global_works[:100])
            )
            st.session_state.context = context
            st.session_state.ran_query = True
            st.rerun() # Rerun to update the chat interface state

# --- Function to Generate PDF ---
def generate_pdf(summary_text, filename="summary.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Research Summary", ln=True, align="C")
    pdf.ln(10)  # Add a line break

    # Add summary content
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, txt=line)
    
    # Save the PDF
    pdf.output(filename)

    # Encode the PDF to make it downloadable
    with open(filename, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    pdf_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{filename}">Download Summary as PDF</a>'
    return pdf_link

# --- Automatically Generate Summary ---
if st.session_state.get("context") and model:
    summary_prompt = f"""
    You are an experienced consultant specialized in advising higher education institutions in the best research strategies to improve their research impact, with special focus on research collaboration. Provide a concise summary comparing the research output of the institution '{institution_name}' on the topic '{topic}' 
    with the country-level and global outputs. Highlight the main differences, focusing on the institution's strengths 
    and weaknesses compared to the country and global datasets. Use the following data:

    {st.session_state.context}
    """
    with st.spinner("Generating summary..."):
        try:
            summary_response = model.generate_content(summary_prompt)

            # Display the summary in a collapsible section
            with st.expander("AI Generated Summary (Click to expand)", expanded=False):
                st.markdown(summary_response.text)

                # Generate and display the PDF download link
                pdf_link = generate_pdf(summary_response.text)
                st.markdown(pdf_link, unsafe_allow_html=True)

                # Add a follow-up question for deeper insights
                # st.markdown("Would you like more insights about this?")
                if st.button("Provide more insights"):
                    deeper_prompt = f"""
                    Based on the previous summary, provide deeper insights into the research output of the institution '{institution_name}' 
                    on the topic '{topic}'. Focus on specific areas where the institution excels or lags behind compared to the country 
                    and global datasets. Include actionable recommendations for improvement.
                    """
                    with st.spinner("Generating deeper insights..."):
                        try:
                            deeper_response = model.generate_content(deeper_prompt)
                            st.markdown("### Deeper Insights")
                            st.markdown(deeper_response.text)
                        except Exception as e:
                            st.error(f"An error occurred while generating deeper insights: {e}")

        except Exception as e:
            st.error(f"An error occurred while generating the summary: {e}")

# --- LLM Chat Interface ---
if st.session_state.get('ran_query'):
    with chat_placeholder.container():
        st.header("GenAI Interaction")
        st.write("Data loaded. You can now ask questions about the research landscape.")
        user_prompt = st.chat_input("Ask a question about the research landscape...")
        if user_prompt and model:
            full_prompt = f"""
You are an experienced consultant specialized in advising higher education institutions in the best research strategies to improve their research impact, with special focus on research collaboration. Provide a concise summary comparing the research output of the institution '{institution_name}' on the topic '{topic}' 
with the country-level and global outputs. Using the research data below, answer the user's question.

{st.session_state.context}

Question: {user_prompt}
"""
            
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")