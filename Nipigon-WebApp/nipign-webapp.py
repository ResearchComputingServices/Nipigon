import os, shutil
import base64
import json

from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit.components.v1 as components
import streamlit as st

from PIL import Image
import glob

from pprint import pprint

from ExDocGen.ExtractedDocumentGenerator import ExtractedDocumentGenerator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DOC_GEN_KEY = 'DOC_GEN_KEY'
EXTRACTED_DOC_KEY = 'EXTRACTED_DOC_KEY'
IMAGE_INDEX_CUR_KEY = 'IMAGE_INDEX_CUR_KEY'
IMAGE_INDEX_MAX_KEY = 'IMAGE_INDEX_MAX_KEY'
IMAGE_FILE_PATHS_KEY = 'IMAGE_FILE_PATHS_KEY'

OUTPUT_DIR_PATH = '.output'

if DOC_GEN_KEY not in st.session_state:
    st.session_state[DOC_GEN_KEY] = ExtractedDocumentGenerator(output_path=OUTPUT_DIR_PATH)

if EXTRACTED_DOC_KEY not in st.session_state:
    st.session_state[EXTRACTED_DOC_KEY] = None
    
if IMAGE_INDEX_CUR_KEY not in st.session_state:
    st.session_state[IMAGE_INDEX_CUR_KEY] = 0

if IMAGE_INDEX_MAX_KEY not in st.session_state:
    st.session_state[IMAGE_INDEX_MAX_KEY] = 0

if IMAGE_FILE_PATHS_KEY not in st.session_state:
    st.session_state[IMAGE_FILE_PATHS_KEY] = []

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define all callback functions

def process_pdf_file(pdf_file) -> None:

    doc_gen = st.session_state[DOC_GEN_KEY]

    if pdf_file != None:
        st.session_state[EXTRACTED_DOC_KEY] = doc_gen.extract_from_stream(  pdf_file.getvalue(),
                                                                            output_name='test')
                
        st.session_state[IMAGE_FILE_PATHS_KEY] = glob.glob(OUTPUT_DIR_PATH+'/annotated_images/*.png')
        st.session_state[IMAGE_FILE_PATHS_KEY].sort()
                
        st.session_state[IMAGE_INDEX_MAX_KEY] = len(st.session_state[IMAGE_FILE_PATHS_KEY])-1
    else:
        st.sidebar.error(f'No file selected for processing!')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

def download() -> None:
       
    if st.session_state[EXTRACTED_DOC_KEY] != None:
        
        object_to_download = json.dumps(st.session_state[EXTRACTED_DOC_KEY].get_json_dict())
        b64 = base64.b64encode(object_to_download.encode()).decode()
        download_filename = 'extracted_document.json'
        dl_link = f"""
                    <html>
                    <head>
                    <title>Start Auto Download file</title>
                    <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
                    <script>
                    $('<a href="data:text/csv;base64,{b64}" download="{download_filename}">')[0].click()
                    </script>
                    </head>
                    </html>
                    """
        components.html(dl_link,height=0)
    else:
        st.sidebar.error(f'No file available for download!')
    
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

def clear() -> None:
    
    st.session_state[EXTRACTED_DOC_KEY] = None
    st.session_state[IMAGE_INDEX_CUR_KEY] = 0
    st.session_state[IMAGE_FILE_PATHS_KEY] = []
    st.session_state[IMAGE_INDEX_MAX_KEY] = 0
    
    dir_path = os.path.join(OUTPUT_DIR_PATH+'/annotated_images')
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
               
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

def increase_current_page():
    if st.session_state[IMAGE_INDEX_CUR_KEY] < st.session_state[IMAGE_INDEX_MAX_KEY]:
        st.session_state[IMAGE_INDEX_CUR_KEY] = st.session_state[IMAGE_INDEX_CUR_KEY] + 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

def decrease_current_page():
    if st.session_state[IMAGE_INDEX_CUR_KEY] > 0:
        st.session_state[IMAGE_INDEX_CUR_KEY] = st.session_state[IMAGE_INDEX_CUR_KEY] - 1
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# initialize the page
st.set_page_config(page_title="Intelligent PDF Textraction",
                   layout="wide")

st.sidebar.markdown(" # Introduction ")
st.sidebar.markdown("The Nipigon project performs \'intelligent\' text extraction from highly structured PDF documents.")

st.sidebar.markdown("## PDF File Picker")

uploaded_file = st.sidebar.file_uploader("Choose a file")

st.sidebar.button('Process', on_click=process_pdf_file, args=(uploaded_file,))
st.sidebar.button('Download', on_click=download)
st.sidebar.button('Clear', on_click=clear)


prev, next, trash = st.columns([1,1,15])
prev.button('Prev',on_click=decrease_current_page)
next.button('Next',on_click=increase_current_page)
st.write(f'Current Page { st.session_state[IMAGE_INDEX_CUR_KEY]} of { st.session_state[IMAGE_INDEX_MAX_KEY]}')

col_2, col_3 = st.columns(2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# Column 2 - PDF Page Images and Annotations
col_2.markdown("## PDF Images")

if st.session_state[EXTRACTED_DOC_KEY] != None:
    
    current_page = st.session_state[IMAGE_INDEX_CUR_KEY]
    max_page = st.session_state[IMAGE_INDEX_MAX_KEY]
            
    image_file_path = st.session_state[IMAGE_FILE_PATHS_KEY][current_page]
    
    image = Image.open(image_file_path)
  
    # this gets the location in image coordinates where the click has
    # happened on the image.
    with col_2:
        value =  streamlit_image_coordinates(image)  
        print(value)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Column 3 - Extract text in a JSON viewer
col_3.markdown("## Extracted Text")

if st.session_state[EXTRACTED_DOC_KEY] != None:
    
    extracted_document = st.session_state[EXTRACTED_DOC_KEY]
    current_page = extracted_document.get_page(st.session_state[IMAGE_INDEX_CUR_KEY])
    
    col_3.markdown(current_page.get_labelled_text())
   

