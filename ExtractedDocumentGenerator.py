import os

import numpy as np
import torch

import fitz

from BoundingBox import generate_bounding_boxes
from ExtractedDocument import ExtractedDocument, DocumentPage

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFAULT_MODEL_WEIGHTS_PATH = './weights/best.pt'
DEFAULT_MODEL_LOCATION = 'ultralytics/yolov5'
DEFAULT_MODEL_TYPE = 'custom'

DEFAULT_ROOT_OUTPUT_PATH = '.output/'
DEFAULT_PDF_PAGE_IMAGE_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, 'pdf_page_images')
DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, 'annotated_images')
DEFAULT_LABEL_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, 'labels')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clean_text(original_text : str) -> str:
    """this function returns cleaned version of the input string with new lines and non printable ascii characters removed.

    Args:
        original_text (str): string which required cleaning

    Returns:
        str: cleaned string
    """
    cleaned_text = ''
    
    for char in original_text:
        # replace new line with space
        if ord(char) == 10:
            char = ' '
        
        # only add printable ascii characters
        if ord(char) > 31 and ord(char) < 127:
            cleaned_text += char

    return cleaned_text.strip()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ExtractedDocumentGenerator:
    
    def __init__(   self,
                    path_to_weights = DEFAULT_MODEL_WEIGHTS_PATH,
                    model_path = DEFAULT_MODEL_LOCATION,
                    model_type = DEFAULT_MODEL_TYPE):
        
        self.model = None
        self._load_model(path_to_weights=path_to_weights,
                         model_type=model_type,
                         model_path=model_path)
              
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       
    def _load_model(self,
                    path_to_weights = DEFAULT_MODEL_WEIGHTS_PATH,
                    model_path = DEFAULT_MODEL_LOCATION,
                    model_type = DEFAULT_MODEL_TYPE) -> None:
        
        self.model = torch.hub.load(repo_or_dir=model_path, 
                                    model=model_type,
                                    path=path_to_weights)
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _check_pdf_file_path(self,
                             pdf_file_path : str):
        
        if not os.path.isfile(pdf_file_path):
            raise FileNotFoundError
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def _check_directory_path(self,
                              dir_path : str) -> None:
        
        if not os.path.exists(dir_path):
            try: 
                os.mkdir(dir_path)    
            except OSError as error:  
                raise OSError from error 
     
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_output_directory_paths(self) -> None:
        
        self._check_directory_path(self.output_path)
        self._check_directory_path(self.pdf_image_output_path)    
        self._check_directory_path(self.annoted_image_output_path)    
        self._check_directory_path(self.label_output_path)
           
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def extract(self,
                pdf_file_path : str,
                include_pages = [],
                save_steps = False) -> ExtractedDocument:
        
        # make sure the pdf file exists
        self._check_pdf_file_path(pdf_file_path)
        
        # make sure the output directories exist
        if save_steps:
            self._check_output_directory_paths()
               
        fitz_doc = fitz.open(pdf_file_path)

        extracted_doc = ExtractedDocument(pdf_file_path)

        for page_number, page in enumerate(fitz_doc):
            if page_number in include_pages or len(include_pages) == 0:
    
                # load the page as a numpy.ndarray
                pix = page.get_pixmap()
                page_img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
                              
                # pass the page_img(numpy.ndarray) to the model to get the results
                results = self.model(   page_img,
                                        size=(pix.height, pix.width))
                
                extracted_page = self._extract_text_from_page(  fitz_page=page,
                                                                page_number=page_number,
                                                                labels=results.xyxy[0].cpu().numpy())
        
                extracted_doc.add_page(extracted_page) 
    
        return extracted_doc
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    
    def _extract_text_from_page(self, 
                                fitz_page : fitz.Page,
                                page_number : int,
                                labels : np.array) -> DocumentPage:
        
        bb_list = generate_bounding_boxes(labels)
        
        extracted_page = DocumentPage(page_number)
            
        for bb in bb_list:
            extracted_page.add_text_block(  text=clean_text(fitz_page.get_textbox(bb.get_rect())),
                                            conf=bb.confidence,
                                            label=bb.label)
           
        return extracted_page
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _display_image(self):
        # display annotated image to screen
        # results.print()
        # r_img = results.render()       
        # cv2.imshow("Image", r_img[0])
        # cv2.waitKey(0)  
        pass
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _get_bounding_boxes_from_np_array(  self,
                                            labels : np.array,
                                            sort_boxes = True,
                                            clean_boxes = True) -> list:              
        bounding_boxes = []

        # labels = [ labelled_bb ]
        # labelled_bb  = [xmin, ymin, xmax, ymax, confidence, class]
        for labelled_bb in labels:
            bounding_boxes.append(self._generate_bounding_boxes(labelled_bb))
       
        if sort_boxes:
            bounding_boxes.sort()

        if clean_boxes:
            bounding_boxes = self._clean_boxes(bounding_boxes)
        
        return bounding_boxes