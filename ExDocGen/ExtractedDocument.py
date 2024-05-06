import json

from dataclasses import dataclass
from pprint import pprint

import pysbd

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

    def __str__(self):
        return self.get_text()

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

        self.document_pages = []
        self.current_page_num = 0

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

    def get_json_dict(self) -> dict:
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

    @property
    def num_pages(self) -> int:
        """returns the number of elements in the document_pages list

        Returns:
            int: # of elements in self.document_pages
        """
        return len(self.document_pages)



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def display(self) -> None:
        """Display the contents of the ExtractedDocument to the screen
        """
        pprint(self.get_json_dict())

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
            json.dump(self.get_json_dict(), final)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~