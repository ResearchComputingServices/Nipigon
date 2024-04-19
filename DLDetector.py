import os
import logging
import uuid

import fitz

from yolov5.detect import run as yolov5_run

from DLDetector_config import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_object_detection(source_path : str,
                         output_path : str,
                         output_name : str,
                         image_size = (792,612),
                         nosave=True):
    
    weights = MODEL_WEIGHTS_PATH   # fine tuned weights
    
    if output_name == None:
        output_name = str(uuid.uuid4())

    yolov5_run( weights=weights,
                source=source_path,
                project=output_path,
                name=output_name,
                imgsz=image_size,
                nosave = nosave,
                line_thickness=1,
                save_txt=True,
                save_conf=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_pdf_page_image(page : fitz.Page, 
                        output_dir_path : str, 
                        page_number : int,
                        image_format = PNG_FORMAT) -> None:

    img = page.get_pixmap()

    output_file_name = f'page-image-{page_number}.{image_format}'
    output_file_path = os.path.join(output_dir_path, output_file_name)

    try:
        img.save(output_file_path)
    except fitz.mupdf.FzErrorSystem:
        logging.error(f'Could not save to directory: {output_file_path}')
        return
    except ValueError:
        logging.error(f'Unknown image format: {image_format}')
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_pdf_document_images(   pdf_file_path : str,
                                output_dir_path : str) -> None:

    try:
        fitz_doc = fitz.open(pdf_file_path)

        for page_number, page in enumerate(fitz_doc):
            save_pdf_page_image(page, output_dir_path, page_number)
    
    except fitz.FileNotFoundError:
        logging.error(f'Could not open file: {pdf_file_path}')
        return

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def perform_document_layout_analysis(pdf_file_path : str,
                                     image_output_path : str,
                                     label_file_output_path : str,
                                     output_name = None):
   
    save_pdf_document_images(pdf_file_path, 
                             image_output_path)

    run_object_detection(source_path=image_output_path,
                         output_path=label_file_output_path,
                         output_name=output_name)
    
