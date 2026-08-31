from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from .schema import FrameAnnotation

def write_contact_sheets(items: list[FrameAnnotation], output_dir: Path, *, columns: int=5, cell_size: tuple[int,int]=(320,180)) -> list[Path]:
    output_dir.mkdir(parents=True,exist_ok=True); outputs=[]
    for start in range(0,len(items),columns*4):
        cells=[]
        for item in items[start:start+columns*4]:
            image=cv2.imread(item.image_path); image=cv2.resize(image,cell_size) if image is not None else np.zeros((cell_size[1],cell_size[0],3),np.uint8)
            cv2.rectangle(image,(0,0),(cell_size[0],22),(0,0,0),-1); cv2.putText(image,f"{item.frame_id} {item.review_status}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
            cells.append(image)
        while len(cells)<columns*4: cells.append(np.zeros((cell_size[1],cell_size[0],3),np.uint8))
        sheet=cv2.vconcat([cv2.hconcat(cells[i:i+columns]) for i in range(0,columns*4,columns)]); path=output_dir/f"contact_{start//(columns*4)+1:03d}.jpg"; cv2.imwrite(str(path),sheet); outputs.append(path)
    return outputs
