import os
import logging

import fitz

from pdf_textractor_config import *
from BoundingBoxGenerator import BoundingBoxGenerator as BBGen
from ExtractedDocument import ExtractedDocument, DocumentPage
from DLDetector import perform_document_layout_analysis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extract_text_from_page(fitz_page : fitz.Page,
                           labels_file : str,
                           page_number : int):
    
    bb_generator = BBGen(   fitz_page.rect.x1, 
                            fitz_page.rect.y1, 
                            LABEL_DICT)
    
    bb_list = bb_generator.get_bounding_boxes_from_file(labels_file)
        
    extracted_page = DocumentPage(page_number)
            
    for bb in bb_list:
        if bb.label not in EXTRACT_LABELS:
            continue
        
        extracted_page.add_text_block(  text=clean_text(fitz_page.get_textbox(bb.get_rect())),
                                        conf=bb.confidence,
                                        label=bb.label)

    return extracted_page
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fitz_extract_text(fitz_document : fitz.Document,
                      label_files : list,
                      extracted_document : ExtractedDocument):
         
    for i,page in enumerate(fitz_document):
        extracted_page = extract_text_from_page(fitz_page=page,
                                                page_number=i,
                                                labels_file=label_files[i])
        
        extracted_document.add_page(extracted_page)      
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pdf_extract_text(pdf_file_path : str,
                     label_files : list) -> ExtractedDocument:
    
    fitz_document = fitz.open(pdf_file_path)
    extracted_document = ExtractedDocument(pdf_file_path)
    
    if len(fitz_document) != len(label_files):
        logging.error(f'# of pages in pdf ({len(fitz_document)}) does not equal # of label files ({len(label_files)})')    
    else:
        fitz_extract_text(fitz_document, 
                          label_files, 
                          extracted_document) 
    
    return extracted_document

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pdf_textractor( pdf_file_path : str,
                    image_output_path : str,
                    label_file_output_path : str)-> ExtractedDocument:
    
    if not os.path.isfile(pdf_file_path):
        logging.error(f'PDF not found at {pdf_file_path}')
        return

    if not os.path.exists(image_output_path):
        logging.warning(f'Image output path not found {image_output_path}. Generating.')

    if not os.path.exists(label_file_output_path):
        logging.warning(f'Image output path not found {label_file_output_path}. Generating.')

    perform_document_layout_analysis(pdf_file_path=pdf_file_path,
                                     image_output_path=image_output_path,
                                     label_file_output_path=label_file_output_path)
    
    return pdf_extract_text(pdf_file_path=pdf_file_path,
                            label_file_output_path=label_file_output_path)
