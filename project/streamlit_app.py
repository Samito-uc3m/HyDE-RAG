import time

import streamlit as st


# --------------- Database Loading Logic ---------------
def load_database():
    """
    Mock function to simulate database loading.
    Replace this with your actual database loading logic.
    """
    st.session_state.db_loaded = True
    st.session_state.show_success = True  # Trigger success message


def main():
    st.title("RAG-based Research Assistant")

    # Initialize session state variables
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "db_loaded" not in st.session_state:
        st.session_state.db_loaded = False
    if "show_success" not in st.session_state:
        st.session_state.show_success = False

    # Reset the success message state if not triggered by "Load Database"
    if st.session_state.show_success and not st.session_state.db_loaded:
        st.session_state.show_success = False

    # -------------------- USER INPUT --------------------
    user_query = st.text_area("Enter your line of investigation:")

    # If the user typed a new query, clear any previous response
    if user_query != st.session_state.last_query:
        st.session_state.response = ""
        st.session_state.last_query = user_query

    # ------------------- SEARCH BUTTON ------------------
    if st.button("Search"):
        if user_query.strip():
            # Call the RAG pipeline
            result = "result"
            st.session_state.response = result
        else:
            st.write("Please enter a query.")

    # ------------------ DISPLAY RESPONSE ----------------
    st.markdown("### RAG Answer")
    st.write(st.session_state.response)

    # ------------------- LOAD DATABASE BUTTON -------------------
    # Use Streamlit's columns for alignment
    _, col2 = st.columns([7, 2])  # Adjust column ratio for alignment

    with col2:
        if st.button(
            "Load Database",
            disabled=st.session_state.db_loaded,
            key="load_db_button",
        ):
            load_database()  # Load the database when button is clicked

    # ------------------ SUCCESS MESSAGE ------------------
    success_placeholder = st.empty()  # Create a placeholder for the success message

    if st.session_state.show_success:
        success_placeholder.success("Database successfully loaded!")
        time.sleep(3)  # Wait for 3 seconds
        success_placeholder.empty()  # Clear the message
        st.session_state.show_success = False


if __name__ == "__main__":
    main()
