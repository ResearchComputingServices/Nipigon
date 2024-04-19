import logging
import argparse

from DLDetector_config import *
from DLDetector import perform_document_layout_analysis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_cli(args_dict : dict) -> None:

    perform_document_layout_analysis(   pdf_file_path=args_dict[CL_ARG_PDF_FLAG],
                                        image_output_path=args_dict[CL_ARG_IMG_FLAG],
                                        label_file_output_path=args_dict[CL_ARG_LABEL_FLAG])
    
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

    parser.add_argument('--'+CL_ARG_IMG_FLAG,
                        default=CL_ARG_IMG_DEFUALT,
                        help = CL_ARG_IMG_HELP)
    
    parser.add_argument('--'+CL_ARG_LABEL_FLAG,
                        default=CL_ARG_LABEL_DEFUALT,
                        help = CL_ARG_LABEL_HELP)

    args = parser.parse_args()

    return vars(args)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# python DLDetector_cli.py --pdf data/sample.pdf --img data/page_images/ --labels data/label_files/
def main():

    args_dict = extract_command_line_args()
    
    logging.basicConfig(filename='DLDetector.log', 
                        encoding='utf-8', 
                        level=logging.DEBUG)
    
    run_cli(args_dict)    
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()