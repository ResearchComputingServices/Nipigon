import os
import json

import torch

import numpy as np

from dataclasses import dataclass
from pprint import pprint

import pysbd

import fitz

from BoundingBoxGenerator import BoundingBoxGenerator as BBGen

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFAULT_ROOT_OUTPUT_PATH = '.output/'
DEFAULT_PDF_PAGE_IMAGE_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, 'pdf_page_images')
DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, 'annotated_images')
DEFAULT_LABEL_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, 'labels')

MODEL_WEIGHTS_PATH = './weights/best.pt'

LABEL_DICT = {  '0':  'Caption',
                '1':  'Footnote',
                '2':  'Formula',
                '3':  'List-item',
                '4':  'Page-footer',
                '5':  'Page-header',
                '6':  'Picture',
                '7':  'Section-header',
                '8':  'Table',
                '9':  'Text',
                '10': 'Title'}

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class DocumentTextBlock:
    """
    Text_block extracted from pdf using yolov5 bounding box and fitz
    """

    def __init__(self,
                 text : str,
                 conf = 0.,
                 label = 'UNKNOWN'):
                
        self.sentences = self._split_sentences(text)
        self.conf = float(conf)
        self.label = label

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __str__(self):
        return f'conf: {self.conf} label: {self.label}\n' + self.text
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    @property
    def text(self) -> str:
        return ('\n'.join(self.sentences)).strip()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _split_sentences(self,
                         text) -> list:
        seg = pysbd.Segmenter(language='en', clean=False)
        return seg.segment(text)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def to_dict(self) -> dict:
        """returns the contains of the object in dictionary form

        Returns:
            dict: dictionary containing all the data of the object
        """
        
        json_dict =  {  'conf' : self.conf,
                        'label' : self.label,
                        'sentences' : []}
        
        for sentence in self.sentences:
            json_dict['sentences'].append(sentence)
        
        return json_dict

# =============================================================================

@dataclass
class DocumentPage:
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, 
                 page_number : int):
               
        self.page_number = page_number
        
        self.current_block_num = 0
        self.document_text_blocks = []
    
     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def __iter__(self):
        self.current_block_num = 0
        return self

    def __next__(self):
        if self.current_block_num < len(self.document_text_blocks):
            current_page = self.document_text_blocks[self.current_block_num]
            self.current_block_num += 1
            return current_page
        else:
            raise StopIteration
                
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    @property
    def num_text_blocks(self) -> int:
        """returns the number of elements in the document_text_blocks list

        Returns:
            int: # of elements in self.document_text_blocks
        """
        return len(self.document_text_blocks)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_text_block( self,
                        text : str,
                        conf = 0.,
                        label = 'UNKNOWN') -> None:
        """Add a text block to the document

        Args:
            text_block (DocumentTextBlock): text block to be added
        """

        self.document_text_blocks.append(DocumentTextBlock(text, conf, label))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_text(self) -> str:
        page_text = ''
        
        for text_block in self.document_text_blocks:
            page_text += text_block.text + '\n'    
            
        return page_text
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def to_dict(self) -> dict:
        """returns the contains of the object in dictionary form

        Returns:
            dict: dictionary containing all the data of the object
        """
        
        json_dict =  {  'page_number' : self.page_number,
                        'document_text_blocks' : []}
        
        for text_block in self.document_text_blocks:
            json_dict['document_text_blocks'].append(text_block.to_dict())
        
        return json_dict
    
# =============================================================================

class ExtractedDocument:
    """
    Class containing the text data extracted from a pdf by fitz
    """
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def __init__(self,
                 file_path : str):
        
        self.pdf_file_path = file_path
        
        self.output_path = DEFAULT_ROOT_OUTPUT_PATH        
        self.pdf_image_output_path = DEFAULT_PDF_PAGE_IMAGE_OUTPUT_PATH       
        self.annoted_image_output_path = DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH
        self.label_output_path = DEFAULT_LABEL_OUTPUT_PATH
        
        self.document_pages = []
        self.current_page_num = 0
        
        self.model = None
        self._load_model()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __iter__(self):
        self.current_page_num = 0
        return self

    def __next__(self):
        if self.current_page_num < len(self.document_pages):
            current_page = self.document_pages[self.current_page_num]
            self.current_page_num += 1
            return current_page
        else:
            raise StopIteration
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _check_pdf_file_path(self):
        if not os.path.isfile(self.pdf_file_path):
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
    
    def _get_json_dict(self) -> dict:
        """Convert the DataSet into a json dictionary

        Returns:
            dict: JSON dictionary representation of the DateSet
        """
        json_dict = {'file_path' : self.pdf_file_path,
                     'document_pages' : []}        

        for page in self.document_pages:
            json_dict['document_pages'].append(page.to_dict())

        return json_dict
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
    def _load_model(self,
                    path_to_weights = MODEL_WEIGHTS_PATH) -> None:
        
        self.model = torch.hub.load(repo_or_dir='ultralytics/yolov5', 
                                    model='custom',
                                    path=path_to_weights)

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    @property
    def num_pages(self) -> int:
        """returns the number of elements in the document_pages list

        Returns:
            int: # of elements in self.document_pages
        """
        return len(self.document_pages)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def extract(self,
                include_pages = []) -> None:
        
        # make sure the pdf file exists
        self._check_pdf_file_path()
        
        # make sure the output directories exist
        self._check_output_directory_paths()
           
        model = self.model
    
        fitz_doc = fitz.open(self.pdf_file_path)

        for page_number, page in enumerate(fitz_doc):
            if page_number in include_pages or len(include_pages) == 0:
    
                # load the page as a numpy.ndarray
                pix = page.get_pixmap()
                page_img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
                
                # pass the page_img(numpy.ndarray) to the model to get the results
                results = model(page_img, size=(792,612))
                #self.display_image()       
                        
                # results.xyxy  = [[xmin, ymin, xmax, ymax, confidence, class]]
                extracted_page = self._extract_text_from_page(  fitz_page=page,
                                                                page_number=page_number,
                                                                labels=results.xyxy[0].numpy())
        
                self.add_page(extracted_page) 
               
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    
    def _extract_text_from_page( self, 
                                fitz_page : fitz.Page,
                                page_number : int,
                                labels : np.array) -> DocumentPage:
        
       
        bb_generator = BBGen(   fitz_page.rect.x1, 
                                fitz_page.rect.y1, 
                                LABEL_DICT)
        
        bb_list = bb_generator.get_bounding_boxes_from_np_array(labels)
        
        extracted_page = DocumentPage(page_number)
            
        for bb in bb_list:
            extracted_page.add_text_block(  text=clean_text(fitz_page.get_textbox(bb.get_rect())),
                                            conf=bb.confidence,
                                            label=bb.label)
            
           
        return extracted_page
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def display_image(self):
        # display annotated image to screen
        # results.print()
        # r_img = results.render()       
        # cv2.imshow("Image", r_img[0])
        # cv2.waitKey(0)  
        pass
    
    def display(self) -> None:
        """Display the contents of the ExtractedDocument to the screen
        """
        pprint(self._get_json_dict())
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def get_page(self,
                 requested_page_num : int) -> DocumentPage:
       
        for page in self.document_pages:
            if page.page_number == requested_page_num:
                return page 
        return None
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    
    def add_page(self,
                 page : DocumentPage) -> None:
        
        self.document_pages.append(page)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    def save_as_json(self,
                     file_path : str) -> None:
        
        """Save the DataSet as a JSON file

        Args:
            file_path (str): File path to save DataSet
        """
        with open(file_path, "w+") as final:
            json.dump(self._get_json_dict(), final)
            
    