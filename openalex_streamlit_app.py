# This script is intended to run with Streamlit. To use it locally, install Streamlit first:
# pip install streamlit

try:
    import streamlit as st
    import pandas as pd
    import requests
    import google.generativeai as genai
    import json
    import urllib.parse
except ModuleNotFoundError as e:
    print(f"Missing module: {e.name}. Please install it using pip before running this script.")
    raise SystemExit

# --- Streamlit Layout ---
st.set_page_config(layout="wide")

# Apply custom CSS for a black background
st.markdown("""
<style>
    .stApp {
        background-color: black;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
YOUR_OPENALEX_EMAIL = "destructordemitos@gmail.com"  # Replace with your actual email
ROR_DATA_FILE = "ror-data.csv"


# Load ROR data
@st.cache_data
def load_ror_data():
    return pd.read_csv(ROR_DATA_FILE)

ror_df = load_ror_data()

# Select institution using searchable dropdown
institution_name = st.selectbox("Institution", ror_df["name"].dropna().unique().tolist())

# Lookup corresponding ROR ID
selected_ror_id = ror_df.loc[ror_df["name"] == institution_name, "id"].values[0]

# Now you can use selected_ror_id in the OpenAlex query
st.write(f"Selected ROR ID: {selected_ror_id}")



# Set Gemini API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Streamlit Layout ---
st.title("OpenAlex LLM Explorer")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Filters")
    topic = st.text_input("Topic")
    # FIX: Use multiselect to allow choosing multiple institutions
    selected_institutions = st.multiselect(
        "Select Institutions",
        options=ror_df["name"].dropna().unique().tolist()
    )
    start_year = st.number_input("Starting Year", min_value=1900, max_value=2025, value=2015)
    run_query = st.button("Run Query")

with col2:
    st.header("LLM Interaction")
    chat_placeholder = st.empty()

# --- Helper Functions ---
def search_openalex(query, institution_ids=None, country_code=None, per_page=10, sort_by='cited_by_count', start_year=None):
    base_url = "https://api.openalex.org/works"
    filters = []

    if query:
        # Use a more comprehensive search field
        filters.append(f"title_and_abstract.search:{query}")
    if institution_ids:
        # FIX: Correctly format multiple ROR IDs for the API call
        # Assumes institution_ids are full ROR URLs like https://ror.org/xxxx
        ror_ids = [i.split('/')[-1] for i in institution_ids]
        filters.append(f"institutions.ror:{'|'.join(ror_ids)}")
    if country_code:
        filters.append(f"institutions.country_code:{country_code}")
    if start_year:
        filters.append(f"from_publication_date:{start_year}-01-01")

    params = {
        "sort": f"{sort_by}:desc",
        "per_page": per_page,
        "mailto": YOUR_OPENALEX_EMAIL,
        "filter": ",".join(filters)
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error(f"Failed to fetch from OpenAlex: {response.status_code}")
        return []

# --- Run Query and Prepare Context ---
if run_query:
    # FIX: Handle a list of institutions instead of a single one
    if not selected_institutions:
        st.error("Please select at least one institution.")
    else:
        inst_rows = ror_df[ror_df['name'].isin(selected_institutions)]
        institution_ids = inst_rows['id'].tolist()
        
        # For country comparison, use the country of the first selected institution
        country_code = inst_rows.iloc[0].get('country.country_code', '') if not inst_rows.empty else ''

        with st.spinner("Fetching data from OpenAlex..."):
            # This single call now fetches works from ALL selected institutions
            inst_works = search_openalex(topic, institution_ids=institution_ids, start_year=start_year)
            
            country_works = []
            if country_code:
                country_works = search_openalex(topic, country_code=country_code, start_year=start_year)
            
            global_works = search_openalex(topic, start_year=start_year)

        def format_works(title, works):
            return f"\n\n### {title}\n" + "\n".join([
                f"- {w.get('title', 'No Title')} ({w.get('publication_year', 'N/A')}), Citations: {w.get('cited_by_count', 0)}"
                for w in works])

        context = (
            format_works(f"Top 10 Works - Selected Institutions ({', '.join(selected_institutions)})", inst_works) +
            format_works(f"Top 10 Works - Country ({country_code})", country_works) +
            format_works("Top 10 Works - Global", global_works)
        )

        st.session_state.context = context
        st.success("Data loaded. You can now chat with the LLM.")

# --- LLM Chat ---
if "context" in st.session_state:
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
