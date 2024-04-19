from BoundingBox import BoundingBox

class BoundingBoxGenerator:
    
    def __init__(self,
                 page_width : int,
                 page_height : int,
                 label_dict : dict):
        
        self.page_width = page_width
        self.page_height = page_height
        
        self.label_dict = label_dict

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
    def get_bounding_boxes_from_file(self,
                                     file_path : str,
                                     sort_boxes = True,
                                     clean_boxes = True) -> list:
        try:
            bounding_box_file = open(file_path,'r',encoding='utf-8')
            
            bounding_boxes = self._generate_bounding_boxes(bounding_box_file.readlines())

            if sort_boxes:
                bounding_boxes.sort()

            if clean_boxes:
                bounding_boxes = self._clean_boxes(bounding_boxes)
        
        except FileNotFoundError:
            print(f'label file not found: {file_path}')

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
                                 file_lines : list) -> list:
    
        bounding_boxes = []

        for line in file_lines:
            x0,y0,x1,y1,label,conf = self._read_bounding_box_line(line)
            bounding_boxes.append(BoundingBox(x0,y0,x1,y1,label,conf))

        return bounding_boxes

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