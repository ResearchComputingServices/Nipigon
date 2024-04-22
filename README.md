# Introduction
The Nipigon project performs "intelligent" text extraction from highly structured PDF documents. Traditional text extraction from PDFs often mixes paragraphs with other text such as section headings, list items, or headers/footers. These issues are most pressing when extracting text from highly structure documents for example the page shown in Figure 1. ![]()


The inclusion of such errors produces extracted text that is difficult to read and requires cleaning before use in downstream analysis steps such as when working with Large Language Models. To prevent this the Nipigon project uses a fine-tuned verison of the Yolov5 object recognition model to first label sections



layout analysis to Text extraction from PDFs with intelligent document layout analysis
