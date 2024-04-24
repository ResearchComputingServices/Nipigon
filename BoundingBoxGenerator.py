from BoundingBox import BoundingBox
import numpy as np

class BoundingBoxGenerator:
    
    def __init__(self,
                 page_width : int,
                 page_height : int,
                 label_dict : dict):
        
        self.page_width = page_width
        self.page_height = page_height
        
        self.label_dict = label_dict

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
    def get_bounding_boxes_from_np_array(   self,
                                            labels : np.array,
                                            sort_boxes = True,
                                            clean_boxes = True) -> list:              
        bounding_boxes = []

        for labelled_bb in labels:
            bounding_boxes.append(self._generate_bounding_boxes(labelled_bb))

       
        if sort_boxes:
            bounding_boxes.sort()

        if clean_boxes:
            bounding_boxes = self._clean_boxes(bounding_boxes)
        
        return bounding_boxes
 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _clean_boxes(self, 
                     bounding_boxes : list) -> list:
        
        cleaned_boxes = []
        
        for i, box_a in enumerate(bounding_boxes):
            
            keep_box = True
            
            for j, box_b in enumerate(bounding_boxes):
                if i == j:
                    continue
                
                if box_a.overlaps(box_b) and box_b.confidence > box_a.confidence:
                    keep_box = False
        
            if keep_box:
                cleaned_boxes.append(box_a)
        
        return cleaned_boxes
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _generate_bounding_boxes(self,
                                 labelled_bounding_box : list) -> list:
    
        x0 = labelled_bounding_box[0]
        y0 = labelled_bounding_box[1]
        x1 = labelled_bounding_box[2]
        y1 = labelled_bounding_box[3]
        conf = labelled_bounding_box[4]
        label = self.label_dict[str(int(labelled_bounding_box[5]))]

        return BoundingBox(x0,y0,x1,y1,label,conf)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
    def _read_bounding_box_line(self,
                                line) -> tuple:
        
        (label_key,x0,y0,w,h,conf) = line.split(' ')
        x_c = float(x0)
        y_c = float(y0)
        half_w = float(w)
        half_h = float(h)
        
        if label_key in self.label_dict.keys():
            label = self.label_dict[label_key]
        else:
            label = 'UNKNOWN'
                            
        x0 = int(self.page_width*(x_c-0.5*half_w))
        x1 = int(self.page_width*(x_c+0.5*half_w))
        
        y0 = int(self.page_height*(y_c-0.5*half_h))
        y1 = int(self.page_height*(y_c+0.5*half_h))
                
        return x0,y0,x1,y1,label,float(conf)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~