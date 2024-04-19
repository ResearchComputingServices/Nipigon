PROG_NAME = 'DocumentLayoutDetector'
PROG_DESC = 'This program uses a fined-tuned version of Yolov5 to perform document layout analysis'

CL_ARG_PDF_FLAG = 'pdf'
CL_ARG_PDF_DEFUALT = './data/sample.pdf'
CL_ARG_PDF_HELP = 'File path to pdf being analyzed.'

CL_ARG_IMG_FLAG = 'img'
CL_ARG_IMG_DEFUALT = './data/page_images'
CL_ARG_IMG_HELP = 'Directory path to store images.'

CL_ARG_LABEL_FLAG = 'labels'
CL_ARG_LABEL_DEFUALT = './data/labels'
CL_ARG_LABEL_HELP = 'Directory path to save label data.'

MODEL_WEIGHTS_PATH = './weights/best.pt'

PNG_FORMAT = 'png'