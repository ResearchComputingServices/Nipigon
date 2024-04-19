# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

EXTRACT_LABELS = ['Section-header', 'Text']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROG_NAME = 'pdfTextractor'

PROG_DESC = """This program use a specially trained version of the Yolov5 object
                recognition model to perform document layout analysis."""

CL_ARG_PDF_FLAG = 'pdf'
CL_ARG_PDF_DEFUALT = './pdfs/'
CL_ARG_PDF_HELP = 'File path to PDF being extracted'

CL_ARG_LABELS_FLAG = 'labels'
CL_ARG_LABELS_DEFUALT = './labels'
CL_ARG_LABELS_HELP = 'Directory path to txt files containing bounding box data in text files'

CL_ARG_OUT_FLAG = 'out'
CL_ARG_OUT_DEFUALT = 'output.json'
CL_ARG_OUT_HELP = 'Filename for saving output'

TXT_FILES = '*.txt'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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