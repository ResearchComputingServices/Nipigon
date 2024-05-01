from ExtractedDocumentGenerator import ExtractedDocumentGenerator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():

    pdf_file_path = 'data/sample.pdf'

    doc_gen = ExtractedDocumentGenerator()
    extracted_doc = doc_gen.extract(pdf_file_path=pdf_file_path,
                                    include_pages=[],
                                    save_steps=True)

    print(f'The file {pdf_file_path} has {extracted_doc.num_pages} pages in it')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()