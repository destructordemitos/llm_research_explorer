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

# Load the topics data from openalex-topics.csv
topics_df = pd.read_csv("openalex-topics.csv")

# Combine the values from topic_name, subfield_name, and field_name into a single list of unique options
topic_options = sorted(
    set(topics_df["topic_name"].dropna().tolist() +
        topics_df["subfield_name"].dropna().tolist() +
        topics_df["field_name"].dropna().tolist())
)

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

    .rounded-box {
    border: 2px solid #4CAF50; /* A green border */
    border-radius: 15px; /* More rounded corners */
    padding: 25px;
    margin-bottom: 20px;
    background-color: #f0fff0; /* A light green background */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2); /* A subtle shadow */
}

    </style>
    """,
    unsafe_allow_html=True
)

# --- Add Logo ---
st.image("images/altusnexus_logo.png", width=200)

# --- Streamlit Layout ---
st.title("AltNex Research Intelligence")
st.write(
    "This app allows you to query research data in granular topics from OpenAlex and interact with it using a large language model (LLM). "
    "By default, the model will select the top 100 works based on citation count for the selected insitutions, the relevant country and global output. You can adjust the filters to explore different aspects of the research landscape."
)

# --- Filters Section ---
st.subheader("Filters")

# First row: Topic and Start Year
col1, col2 = st.columns([7, 3])  # Adjusted column widths: 70% for col1, 30% for col2
with col1:
    topic = st.selectbox(
        "Topic",
        options=[""] + topic_options,  # Add an empty option for default
        format_func=lambda x: "Select a topic" if x == "" else x,
        key="topic_filter"
    )
with col2:
    start_year = st.number_input("Start Year", min_value=1980, max_value=2025, value=2020, step=1, key="year_input")

# Second row: Country Filter
countries = sorted(ror_df["country.country_name"].dropna().unique().tolist())  # Extract unique country names
selected_countries = st.multiselect(
    "Select Countries (multiple allowed)",
    options=countries,
    default=[],
    key="country_filter"
)

# Third row: Institution Filter (Multiple Selections Allowed)
if selected_countries:
    filtered_ror_df = ror_df[ror_df["country.country_name"].isin(selected_countries)]
else:
    filtered_ror_df = ror_df

selected_institutions = st.multiselect(
    "Select Institutions (up to 5)",
    options=sorted(filtered_ror_df["name"].dropna().unique().tolist()),
    default=[],
    key="institution_filter"
)

# Validate the number of selected institutions
if len(selected_institutions) > 5:
    st.error("You can select up to 5 institutions only.")
    selected_institutions = selected_institutions[:5]  # Limit to 5 institutions

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
def search_openalex(query, institution_ids=None, country_code=None, per_page=100, sort_by='cited_by_count', start_year=None):
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
        "mailto": st.secrets["openalex_email"],
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
            f"Institution: {w.get('institution_name', 'N/A')}",
            f"Title: {w.get('title', 'No Title')}",
            f"Publication Year: {w.get('publication_year', 'N/A')}",
            f"Citations: {w.get('cited_by_count', 0)}",
            f"Authors: {w.get('authors_list', 'N/A')}",
            f"Institutions: {w.get('institutions_list', 'N/A')}",
            f"Abstract: {w.get('abstract', 'N/A')[:200]}..."
        ]
        fwci_value = w.get('fwci')
        if fwci_value is not None:
            output_parts.append(f"FWCI: {fwci_value:.2f}" if isinstance(fwci_value, (int, float)) else f"FWCI: {fwci_value}")
        formatted_output += ", ".join(output_parts) + "\n"
    return formatted_output


# --- Streamlit UI ---
# --- Main Logic ---
if run_query:
    if not all([topic, start_year]):
        st.error("Please fill in all required fields before running the query.")
    elif not selected_institutions:
        st.error("Please select at least one institution.")
    else:
        with st.spinner("Fetching data from OpenAlex..."):
            # Fetch records for each selected institution
            combined_records = []
            for institution in selected_institutions:
                inst_row = ror_df[ror_df['name'] == institution]
                if inst_row.empty:
                    st.warning(f"Lookup failed: The institution '{institution}' was not found in the CSV.")
                    continue
                institution_id = inst_row.iloc[0]['id']
                institution_records = search_openalex(topic, institution_ids=[institution_id], start_year=start_year)
                for record in institution_records:
                    record["institution_name"] = institution  # Add institution name to each record
                combined_records.extend(institution_records)

            # Convert combined records to a DataFrame
            if combined_records:
                combined_df = pd.DataFrame(combined_records)
                combined_df["abstract"] = combined_df["abstract_inverted_index"].apply(reconstruct_abstract)
                combined_df["authors_list"], combined_df["institutions_list"] = zip(
                    *combined_df["authorships"].apply(extract_authors_and_institutions)
                )
            else:
                st.error("No records were found for the selected institutions.")
                combined_df = pd.DataFrame()

            # --- Results Section ---
            if not combined_df.empty:
                st.markdown('<div class="results-section">', unsafe_allow_html=True)
                st.subheader("Results")
                st.dataframe(combined_df[["institution_name", "title", "publication_year", "authors_list", "institutions_list", "cited_by_count", "fwci", "abstract"]])
                st.markdown("</div>", unsafe_allow_html=True)

                # --- Fetch Country and Global Records ---
                country_code = None
                if selected_countries:
                    country_row = ror_df[ror_df["country.country_name"] == selected_countries[0]]
                    if not country_row.empty:
                        country_code = country_row.iloc[0]["country.country_code"]

                country_records = search_openalex(topic, country_code=country_code, start_year=start_year)
                global_records = search_openalex(topic, start_year=start_year)

                # --- Post-process country_records ---
                for record in country_records:
                    authors, institutions = extract_authors_and_institutions(record.get("authorships"))
                    record["authors_list"] = authors
                    record["institutions_list"] = institutions
                    record["institution_name"] = institutions.split(",")[0] if institutions != "N/A" else "N/A"
                    record["abstract"] = reconstruct_abstract(record.get("abstract_inverted_index"))

                # --- Post-process global_records ---
                for record in global_records:
                    authors, institutions = extract_authors_and_institutions(record.get("authorships"))
                    record["authors_list"] = authors
                    record["institutions_list"] = institutions
                    record["institution_name"] = institutions.split(",")[0] if institutions != "N/A" else "N/A"
                    record["abstract"] = reconstruct_abstract(record.get("abstract_inverted_index"))

                # --- Prepare Context for LLM ---
                country_context = ""
                for country in selected_countries:
                    country_row = ror_df[ror_df["country.country_name"] == country]
                    if not country_row.empty:
                        country_code = country_row.iloc[0]["country.country_code"]
                        country_records_loop = search_openalex(topic, country_code=country_code, start_year=start_year)
                        # Post-process these records too
                        for record in country_records_loop:
                            authors, institutions = extract_authors_and_institutions(record.get("authorships"))
                            record["authors_list"] = authors
                            record["institutions_list"] = institutions
                            record["institution_name"] = institutions.split(",")[0] if institutions != "N/A" else "N/A"
                            record["abstract"] = reconstruct_abstract(record.get("abstract_inverted_index"))
                        country_context += format_works(f"Country Results: {country}", country_records_loop)

                context = (
                    format_works("Combined Institution Results", combined_records) +
                    country_context +
                    format_works("Global Results", global_records)
                )
                st.session_state.context = context
                st.session_state.ran_query = True
                st.rerun()  # Rerun to update the chat interface state

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
        pdf.multi_cell(0, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'))
    
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
    # Define institution_name based on the selected institutions
    institution_name = ", ".join(selected_institutions) if selected_institutions else "the selected institution(s)"
    
    summary_prompt = f"""
    You are an experienced consultant specialized in advising higher education institutions in the best research strategies to improve their research impact, with special focus on research collaboration. Provide a concise summary comparing the research output of the institution '{selected_institutions}' on the topic '{topic}' 
    with the country-level and global outputs. Consider that the institutions dataframe (combined_df) may contain information for multiple institutions, identified in the institution_name column which you can use to filter their respective publications. Highlight the main differences, focusing on the institutions' strengths 
    and weaknesses compared to the country and global datasets. Never include hypothetical or illustrative information or tables. Use the following data:

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
                    Provide deeper and detailed insights into the research output of the institution '{selected_institutions}' 
                    on the topic '{topic}'. Consider that the institutions dataframe may contain information for multiple institutions. The publications for individual institution can be filtered based in the institution_name column. Focus on specific areas where the institution excels or lags behind compared to the country 
                    and global datasets. If there's sufficient data, include comparative tables based on the available data. Never include illustrative tables or hypothetical information. Always use real data derived from the datasets. Include actionable recommendations for improvement. Use the following data:

    {st.session_state.context}

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
You are an experienced consultant specialized in advising higher education institutions in the best research strategies to improve their research impact, with special focus on research collaboration. Provide a concise summary comparing the research output of the institution '{selected_institutions}' on the topic '{topic}'
with the country-level and global outputs. Consider that the institutions dataframe (combined_df) may contain information for multiple institutions, identified in the institution_name column, which you can use to filter their respective publications. Using the research data below, answer the user's question.

{st.session_state.context}

Question: {user_prompt}
"""
            
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")

# --- Show LLM Context for Debugging ---
with st.expander("Show LLM Context (Debug)", expanded=False):
    if "context" in st.session_state:
        st.markdown(f"<pre>{st.session_state.context}</pre>", unsafe_allow_html=True)
    else:
        st.info("No context available yet. Run a query to generate context.")
