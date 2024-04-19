import os

import argparse
import logging

from glob import glob

from pdf_textractor_config import *
from pdf_textractor import pdf_extract_text

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run(args_dict : dict) -> None:
    """This function checks that the require files are found and then runs 
    the text extraction method.

    Args:
        args_dict (dict): dictionary of command line arguments
    """

    # check that pdf file exists
    pdf_file_path = args_dict[CL_ARG_PDF_FLAG]
    if not os.path.isfile(pdf_file_path):
        logging.error(f'PDF file not found {pdf_file_path}')
        return

    # check that labels directory exists
    labels_directory_path = args_dict[CL_ARG_LABELS_FLAG]
    if not os.path.exists(labels_directory_path):
        logging.error(f'Directory not found {labels_directory_path}')
        return

    # check that label files found
    labels_files = sorted(glob(os.path.join(labels_directory_path,TXT_FILES)))
    if len(labels_files) == 0:
        logging.error(f'No label files found in director {labels_directory_path}')
        return

    # if everything checks out run the extraction method
    extracted_document = pdf_extract_text(pdf_file_path, labels_files)     

    extracted_document.save_as_json(args_dict[CL_ARG_OUT_FLAG])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     
def extract_command_line_args() -> dict:
    """generates the parser and applies it to the command line args

    Returns:
        dict: a dictionary containing the command line args.
    """
    
    parser = argparse.ArgumentParser(prog=PROG_NAME,
                                     description=PROG_DESC)

    parser.add_argument('--'+CL_ARG_PDF_FLAG,
                        default=CL_ARG_PDF_DEFUALT,
                        help = CL_ARG_PDF_HELP)

    parser.add_argument('--'+CL_ARG_OUT_FLAG,
                        default=CL_ARG_OUT_DEFUALT,
                        help=CL_ARG_OUT_HELP)

    parser.add_argument('--'+CL_ARG_LABELS_FLAG,
                        default=CL_ARG_LABELS_DEFUALT,
                        help=CL_ARG_LABELS_HELP)

    args = parser.parse_args()

    return vars(args)
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# python pdf_textractor_cli.py --pdf ./EMN 2020-1.pdf --labels ./data/exp9/labels/ --out output.json

def main():

    args_dict = extract_command_line_args()
    
    logging.basicConfig(filename='pdf_textractor.log', 
                        encoding='utf-8', 
                        level=logging.DEBUG)
    
    run(args_dict)    
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()