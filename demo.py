from util import *
import requests

#paste the image path here 
img_path=r" "

#path for car_part
meta_data_json_path1=" "

#path for damaged_car_parts
meta_data_json_path2=" "

data_car=requests.get(meta_data_json_path1)
data_damaged_part=requests.get(meta_data_json_path2)

jsondata1=data_car.json()
df1=pd.DataFrame.from_dict(jsondata1)
jsondata2=data_damaged_part.json()
df2=pd.DataFrame.from_dict(jsondata2)

pts=[[] for i in range(len(df1))]
pts2=[[] for i in range(len(df2))]

image_list=damageIdentification(img_path=img_path, meta_data_json_path1=meta_data_json_path1, meta_data_json_path2=meta_data_json2))
calculate_percentage_for_damaged_part=calculate_area(pts=pts,pys2=pts2)

print(calculate_percentage_for_damaged_part)

cv2.imshow('Polygoned_img',image_list[0])
cv2.imshow('Rectangle_with_label_img',image_list[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

