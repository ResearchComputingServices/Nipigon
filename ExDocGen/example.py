from ExtractedDocumentGenerator import ExtractedDocumentGenerator

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    
    pdf_file_path = 'data/sample.pdf'
    
    doc_gen = ExtractedDocumentGenerator()
    
    extract_doc = doc_gen.extract(pdf_file_path)

    extract_doc.save_as_json('output.json')

    for page in extract_doc:
        print(page)
        input('press ENTER for next page.')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    main()