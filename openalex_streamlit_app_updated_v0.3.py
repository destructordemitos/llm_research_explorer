import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
import json
import urllib.parse

# --- Configuration ---
YOUR_OPENALEX_EMAIL = "destructordemitos@gmail.com"  # Replace with your actual email
ROR_DATA_FILE = "ror-data.csv"

# --- Streamlit Layout ---
st.set_page_config(layout="wide")

# Load ROR data
@st.cache_data
def load_ror_data():
    return pd.read_csv(ROR_DATA_FILE)

ror_df = load_ror_data()
institution_names = ror_df['name'].dropna().unique().tolist()

# Set Gemini API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


def fetch_openalex_data(filter_query, topic, start_year):
    base_url = "https://api.openalex.org/works"
    filters = []
    if filter_query:
        filters.append(filter_query)
    if topic:
        filters.append(f"title_and_abstract.search:{urllib.parse.quote(topic)}")
    if start_year:
        filters.append(f"from_publication_date:{start_year}-01-01")
    params = {
        "filter": ",".join(filters),
        "per-page": 25,
        "mailto": YOUR_OPENALEX_EMAIL,
    }
    response = requests.get(base_url, params=params)
    return pd.DataFrame(response.json().get("results", []))


# --- Streamlit Layout ---
st.title("Altus Nexus Research Intelligence ")
st.subheader("Explore research data from OpenAlex")
st.write("This app allows you to query research data in granular topics from OpenAlex and interact with it using a large language model (LLM). By default the model will select the top 50 works based on citation count, but you can adjust the filters to explore different aspects of the research landscape.")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Filters")
    topic = st.text_input("Topic")
    start_year = st.number_input("Start Year", min_value=2000, max_value=2025, value=2020, step=1)
    institution_name = st.selectbox(
        "Institution", 
        options=ror_df["name"].dropna().unique().tolist(), 
        format_func=lambda x: "Type and choose an option" if x == "" else x
    )

    run_query = st.button("Run Query")

    if institution_name and topic and start_year:
        pass


with col2:
    st.header("GenAI Interaction")
    st.write("Once you run a query, you can ask questions about the research landscape. The LLM will use the context provided by the research data to answer your questions. The LLM can make mistakes so you should always check the data.")
    chat_placeholder = st.empty()
    results_placeholder = st.empty()


# --- Helper Functions ---
def search_openalex(query, institution_ids=None, country_code=None, per_page=50, sort_by='cited_by_count', start_year=None):
    base_url = "https://api.openalex.org/works"
    
    # Ensure 'authorships' is selected to get author and institution details
    select_fields = (
        "id,title,publication_year,open_access,abstract_inverted_index," 
        "authorships,fwci,cited_by_count,cited_by_percentile_year,"
        "countries_distinct_count,institutions_distinct_count," # institutions_distinct_count is just a count
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
        "per_page": per_page,  # Updated to fetch 50 works
        "mailto": YOUR_OPENALEX_EMAIL,
        "filter": ",".join(filters),
        "select": select_fields
    }

    req = requests.Request('GET', base_url, params=params)
    prepared = req.prepare()
    
    if institution_ids:
        st.info(f"Debugging Institutional API URL:\n{prepared.url}")

    with requests.Session() as s:
        response = s.send(prepared)

    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error(f"Failed to fetch from OpenAlex: {response.status_code} - {response.text}")
        st.text_area("Response content:", response.text)
        return []

# Function to reconstruct abstract from inverted index
def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""
    
    words_with_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            words_with_positions.append((word, pos))
    
    # Sort words by their position
    words_with_positions.sort(key=lambda x: x[1])
    
    # Extract just the words
    abstract_words = [word for word, pos in words_with_positions]
    
    return " ".join(abstract_words)

# Function to extract authors and institutions from authorships data
def extract_authors_and_institutions(authorships):
    authors_list = []
    institutions_list = []
    if authorships:
        for authorship in authorships:
            author_name = authorship.get('author', {}).get('display_name', 'N/A')
            authors_list.append(author_name)
            
            if 'institutions' in authorship and authorship['institutions']:
                for inst in authorship['institutions']:
                    inst_name = inst.get('display_name', 'N/A')
                    if inst_name not in institutions_list: # Avoid duplicates
                        institutions_list.append(inst_name)
    return ", ".join(authors_list), ", ".join(institutions_list)
    
# --- Run Query and Prepare Context ---
if run_query:
    if not all([institution_name, topic, start_year]):
        st.error("Please fill in all filter fields before running the query.")
    else:
        # Extract institution_id from the selected institution name
        inst_row = ror_df[ror_df['name'] == institution_name]

        if inst_row.empty:
            st.error("Lookup failed: The selected institution name was not found in the CSV.")
        else:
            institution_id = inst_row.iloc[0]['id']  # Extract the ROR ID
            country_code = inst_row.iloc[0].get('country.country_code', '')

            with st.spinner("Fetching data from OpenAlex..."):
                inst_works = search_openalex(topic, institution_ids=[institution_id], start_year=start_year)
                country_works = search_openalex(topic, country_code=country_code, start_year=start_year)
                global_works = search_openalex(topic, start_year=start_year)

        # Pre-process abstracts, authors, and institutions for display and LLM context
        for work_list in [inst_works, country_works, global_works]:
            for work in work_list:
                # Reconstruct abstract
                if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
                    work['abstract'] = reconstruct_abstract(work['abstract_inverted_index'])
                else:
                    work['abstract'] = "No abstract available."
                
                # Extract authors and institutions
                authors_str, institutions_str = extract_authors_and_institutions(work.get('authorships'))
                work['authors_list'] = authors_str
                work['institutions_list'] = institutions_str


                # Define the columns to display for all dataframes
                cols_to_display = [
                    "title", 
                    "publication_year", 
                    "authors_list", # New column for authors
                    "institutions_list", # New column for institutions
                    "cited_by_count", 
                    "open_access", 
                    "abstract", 
                    "fwci", 
                    "cited_by_percentile_year", 
                    "countries_distinct_count", 
                    "institutions_distinct_count", # This is still the count, not the list
                    "keywords" 
                ]

                with results_placeholder.container():
                    st.subheader("Institution Results")
                    inst_df = pd.DataFrame(inst_works)
                    st.dataframe(inst_df.loc[:, [col for col in cols_to_display if col in inst_df.columns]])

                    st.subheader("Country-Level Results")
                    country_df = pd.DataFrame(country_works)
                    st.dataframe(country_df.loc[:, [col for col in cols_to_display if col in country_df.columns]])

                    st.subheader("Global Results")
                    global_df = pd.DataFrame(global_works)
                    st.dataframe(global_df.loc[:, [col for col in cols_to_display if col in global_df.columns]])

                def format_works(title, works):
                    formatted_output = f"\n\n### {title}\n"
                    for w in works:
                        output_parts = [
                            f"- Title: {w.get('title', 'No Title')}",
                            f"Publication Year: {w.get('publication_year', 'N/A')}",
                            f"Citations: {w.get('cited_by_count', 0)}"
                        ]
                        
                        if w.get('authors_list'): # Use the pre-processed authors_list
                            output_parts.append(f"Authors: {w['authors_list']}")
                        if w.get('institutions_list'): # Use the pre-processed institutions_list
                            output_parts.append(f"Institutions: {w['institutions_list']}")

                        if w.get('abstract'):
                            output_parts.append(f"Abstract: {w['abstract'][:200]}...")
                        
                        fwci_value = w.get('fwci')
                        if isinstance(fwci_value, (int, float)):
                            output_parts.append(f"FWCI: {fwci_value:.2f}")
                        elif fwci_value is not None:
                            output_parts.append(f"FWCI: {fwci_value}")

                        cited_by_percentile_year_value = w.get('cited_by_percentile_year')
                        if isinstance(cited_by_percentile_year_value, (int, float)):
                            output_parts.append(f"Citation Normalized Percentile: {cited_by_percentile_year_value:.2f}")
                        elif cited_by_percentile_year_value is not None:
                            output_parts.append(f"Citation Normalized Percentile: {cited_by_percentile_year_value}")

                        if w.get('countries_distinct_count') is not None:
                            output_parts.append(f"Distinct Countries: {w['countries_distinct_count']}")
                        if w.get('institutions_distinct_count') is not None:
                            output_parts.append(f"Distinct Institutions (Count): {w['institutions_distinct_count']}") # Clarify this is a count
                        if w.get('keywords'):
                            keyword_names = [k['display_name'] for k in w['keywords'] if 'display_name' in k]
                            if keyword_names:
                                output_parts.append(f"Keywords: {', '.join(keyword_names)}")
                        
                        if w.get('open_access') and w['open_access'].get('is_oa') is not None:
                            output_parts.append(f"Open Access: {'Yes' if w['open_access']['is_oa'] else 'No'}")
                        
                        formatted_output += ", ".join(output_parts) + "\n"
                    return formatted_output

                context = (
                    format_works(f"Top 10 Works - {institution_name}", inst_works) +
                    format_works(f"Top 10 Works - Country ({country_code})", country_works) +
                    format_works("Top 10 Works - Global", global_works)
                )

                st.session_state.context = context
                with chat_placeholder.container():
                    st.success("Data loaded. You can now chat with the LLM.")
# --- LLM Chat ---
if "context" in st.session_state:
    with chat_placeholder.container():
        user_prompt = st.chat_input("Ask a question about the research landscape...")
        if user_prompt:
            full_prompt = f"""
            Using the research data below, answer the user's question.

            {st.session_state.context}

            Question: {user_prompt}
            """

            with st.spinner("Thinking..."):
                response = model.generate_content(full_prompt)
                st.markdown(response.text)