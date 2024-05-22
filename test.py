import json

from ExDocGen.ExtractedDocumentGenerator import ExtractedDocumentGenerator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():

    pdf_file_path = 'data/sample_short.pdf'

    doc_gen = ExtractedDocumentGenerator()
    extracted_doc = doc_gen.extract_from_path(  pdf_file_path=pdf_file_path,
                                                include_pages=[],
                                                output_name='test')

    extracted_doc.save_as_json('output.json')  
    
    print(f'The file {pdf_file_path} has {extracted_doc.num_pages} pages in it')
    for page in extracted_doc:
        print(page.get_text())
        input('Press ENTER to see next page')

      

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()