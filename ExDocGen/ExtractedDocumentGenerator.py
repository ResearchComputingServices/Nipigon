import os
import io

import numpy as np
import torch
from PIL import Image, ImageDraw
import fitz

from .BoundingBox import generate_bounding_boxes
from .ExtractedDocument import ExtractedDocument, DocumentPage
from .Colours import COLOURS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_WEIGHTS_PATH = os.path.join(DIR_PATH,'weights/best.pt')
DEFAULT_MODEL_LOCATION = 'ultralytics/yolov5'
DEFAULT_MODEL_TYPE = 'custom'

DEFAULT_ROOT_OUTPUT_PATH = '.output/'
PDF_IMAGE_DIR_PATH = 'pdf_page_images'
ANNOTATED_IMAGE_PATH = 'annotated_images'
DEFAULT_PDF_PAGE_IMAGE_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, PDF_IMAGE_DIR_PATH)
DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH = os.path.join(DEFAULT_ROOT_OUTPUT_PATH, ANNOTATED_IMAGE_PATH)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def clean_text(original_text : str) -> str:
    """this function returns cleaned version of the input string with new
    lines and non printable ascii characters removed.

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
    """This classes uses the a fine-tuned object detection model to extract
    extract from a pdf in an intelligent way
    """

    def __init__(   self,
                    path_to_weights = DEFAULT_MODEL_WEIGHTS_PATH,
                    model_path = DEFAULT_MODEL_LOCATION,
                    model_type = DEFAULT_MODEL_TYPE,
                    output_path = DEFAULT_ROOT_OUTPUT_PATH):

        self.model = None
        self._load_model(   path_to_weights=path_to_weights,
                            model_type=model_type,
                            model_path=model_path)

        self.output_path = output_path
        self.pdf_image_output_path =  os.path.join(self.output_path, PDF_IMAGE_DIR_PATH)
        self.annoted_image_output_path = os.path.join(self.output_path, ANNOTATED_IMAGE_PATH)
        self._check_output_directory_paths()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _load_model(self,
                    path_to_weights = DEFAULT_MODEL_WEIGHTS_PATH,
                    model_path = DEFAULT_MODEL_LOCATION,
                    model_type = DEFAULT_MODEL_TYPE) -> None:

        self.model = torch.hub.load(repo_or_dir=model_path,
                                    model=model_type,
                                    path=path_to_weights)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_pdf_file_path(   self,
                                pdf_file_path : str):

        if not os.path.isfile(pdf_file_path):
            raise FileNotFoundError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_directory_path(  self,
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _extract(   self,
                    fitz_doc : fitz.Document,
                    include_pages = [],
                    output_name = None) -> ExtractedDocument:
        
        extracted_doc = ExtractedDocument(fitz_doc.name)

        for page_number, page in enumerate(fitz_doc):
            
            if page_number in include_pages or len(include_pages) == 0:

                # load the page as a numpy.ndarray
                pix = page.get_pixmap()
                page_img = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))

                # pass the page_img(numpy.ndarray) to the model to get the results
                results = self.model(   page_img,
                                        size=(792,612))

                extracted_page = self._extract_text_from_page(  fitz_page=page,
                                                                page_number=page_number,
                                                                labels=results.xyxy[0].cpu().numpy())

                extracted_doc.add_page(extracted_page)

                # save the intermediate images if requested
                if output_name != None:
                    self._save_images(  output_name,
                                        page_number,
                                        page_img,
                                        results.xyxy[0].cpu().numpy())

        return extracted_doc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def extract_from_stream(self,
                            pdf_file_stream : io.BytesIO,
                            include_pages = [],
                            output_name = None) -> ExtractedDocument:
        
        fitz_doc = fitz.open('pdf',io.BytesIO(pdf_file_stream))

        return self._extract(fitz_doc=fitz_doc,
                             include_pages=include_pages,
                             output_name=output_name) 


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def extract_from_path(  self,
                            pdf_file_path : str,
                            include_pages = [],
                            output_name = None) -> ExtractedDocument:
        """_summary_

        Args:
            pdf_file_path (str): _description_
            include_pages (list, optional): _description_. Defaults to [].
            save_steps (bool, optional): _description_. Defaults to False.

        Returns:
            ExtractedDocument: _description_
        """

        # make sure the pdf file exists
        self._check_pdf_file_path(pdf_file_path)
        fitz_doc = fitz.open(pdf_file_path)

        return self._extract(fitz_doc=fitz_doc,
                             include_pages=include_pages,
                             output_name=output_name)        

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _save_images(   self,
                        pdf_file_path : str,
                        page_number : int,
                        page_img : np.array,
                        labels : np.array) -> None:

        page_image_filename = self._get_page_image_file_name(pdf_file_path, page_number)
        self._save_page_image(  page_image_filename,
                                page_img)

        annotated_image_filename = self._get_annotated_image_file_name(pdf_file_path, page_number)
        self._save_annotated_image( annotated_image_filename,
                                    page_img,
                                    labels)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _save_annotated_image(  self,
                                annotated_image_filename,
                                page_img,
                                labels) -> None:

        img = Image.fromarray(page_img)
        img1 = ImageDraw.Draw(img)
        for label in labels:
            x0 = label[0]
            y0 = label[1]
            x1 = label[2]
            y1 = label[3]
            color = COLOURS[int(label[5])]
            img1.rectangle([(x0,y0),(x1,y1)], outline = color)

        annotated_image_path = os.path.join(self.annoted_image_output_path, 
                                            annotated_image_filename)
        img.save(annotated_image_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _save_page_image(self,
         page_image_filename,
         page_img) -> None:

        page_image_path = os.path.join(self.pdf_image_output_path, page_image_filename)
        Image.fromarray(page_img).save(page_image_path)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_page_number_str(self, 
                             page_number : int) -> str:

        page_number_str = ''
        
        if page_number < 10:
            page_number_str = '00'+str(page_number_str)
        elif page_number < 100:
            page_number_str = '0'+str(page_number_str)
        else:
            page_number_str = (page_number_str)
        
        return page_number_str

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_page_image_file_name(  self,
                                    pdf_file_path : str,
                                    page_number : int) -> str:
        pdf_file_name = pdf_file_path.split('/')[-1].split('.')[0]

        page_image_file_name = pdf_file_name+'_page_'+self._get_page_number_str(page_number)+'_image.png'

        return page_image_file_name
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_annotated_image_file_name( self,
                                        pdf_file_path : str,
                                        page_number : int) -> str:

        pdf_file_name = pdf_file_path.split('/')[-1].split('.')[0]

        annotated_image_file_name = pdf_file_name+'_page_'+self._get_page_number_str(page_number)+'_annotated.png'

        return annotated_image_file_name
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
