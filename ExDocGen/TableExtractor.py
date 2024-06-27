
import easyocr
import torch
from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
from PIL import Image, ImageDraw
import numpy as np

from pprint import pprint

 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rescale_bboxes(out_bbox, size):
    img_h, img_w, _ = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def outputs_to_objects(outputs, img_size, id2label):
    
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TableExtractor:

    def __init__(self):
        self.table_transformer = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
        self.reader = easyocr.Reader(['en']) 
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    def extract_table(self, 
                      image):
        
        feature_extractor = DetrFeatureExtractor()
        encoding = feature_extractor(image, return_tensors="pt")
        
        with torch.no_grad():
            tables = self.table_transformer(**encoding)

        structure_id2label = self.table_transformer.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"
        
        table = outputs_to_objects(tables, 
                                   image.shape, 
                                   structure_id2label)
        
        self._annotate_image(image, table)
        
        cell_coordinates = get_cell_coordinates_by_row(table)
        
        img = Image.fromarray(image)
        
        for row in cell_coordinates:
            row_text = []
            for cell in row["cells"]:
                cell_bbox = cell["cell"]
                cell_text = self.read_text_from_rectangle(img, cell_bbox)
                row_text.append(cell_text)
            
            pprint(row_text)
            input('Press ENTER to see next row')
            
        return cells
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def read_text_from_rectangle(self,
                                 image, 
                                 rectangle):        
                
        # Crop the image to the specified rectangle
        # Rectangle format: (x_min, y_min, x_max, y_max)
        cropped_image = image.crop(rectangle)
                
        # Read text from the cropped image
        results = self.reader.readtext(np.array(cropped_image))
        
        # Extracting text from the results
        text_list = [result[1] for result in results]
        
        # Joining the list of texts into a single string
        text = ' '.join(text_list)
        
        return text
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def _annotate_image(self, 
                        image, 
                        table):
        
        colours =   {   'table column' :(255,0,0), 
                        'table row' : (0,255,0), 
                        'table column header' : (0,0,255), 
                        'table' : (255,0,255)}
        
        img = Image.fromarray(image)
        img1 = ImageDraw.Draw(img)
        
        for item in table:
            
            if item['label'] != 'table column':
                continue
            
            bb = item['bbox']
            
            x0 = bb[0]
            y0 = bb[1]
            x1 = bb[2]
            y1 = bb[3]
            color = colours[item['label']]
                
            img1.rectangle([(x0,y0),(x1,y1)], outline = color)

        img.save('annotated_image.png')
        
     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~