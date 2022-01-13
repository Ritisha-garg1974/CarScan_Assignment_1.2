#importing important libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import math
import cv2

#functions

def damageIdentification(img_path,meta_data_json_path1,meta_data_json_path2):
    #loading the dataset for car_parts,damaged_parts and loading the required image 
    data_car=requests.get(meta_data_json_path1)
    data_damaged_part=requests.get(meta_data_json_path2)
    img=cv2.imread(img_path)
    result=[]
    
    #converting json data into a dataframe for car_parts.
    jsondata1=data_car.json()
    df1=pd.DataFrame.from_dict(jsondata1)
    
    #converting json data into a dataframe for damaged_car_parts.
    jsondata2=data_damaged_part.json()
    df2=pd.DataFrame.from_dict(jsondata2)
        
    #function for denormalizing the points by formuala :-normalized_x=(x/image_width)*100, normalized_y=(y/image_height)*100
    def denormalize(pts,image_width,image_height):
        denormalized_x=(pts[0]*image_width)/100
        denormalized_y=(pts[1]*image_height)/100
        return (denormalized_x,denormalized_y)
    
    #denormalizing the points for car_parts
    for i in range(0,len(df1)):
        for key in df1['value'][i]:
            if key=='points':
                for j in range(len(df1['value'][i][key])):
                    df1['value'][i][key][j]=denormalize(df1['value'][i][key][j],df1['original_width'][i],df1['original_height'][i])

    #Dividing the car_points inot 2 different arrays(pts,polygonlabel) correspondingly for simplicity.
    import math
    pts=[[] for i in range(len(df1))]
    polygonlabel=[]
    for i in range(0,len(df1)):
        for key in df1['value'][i]:
            if key=='points':
                for j in range(len(df1['value'][i][key])):
                    pts[i].append((math.floor(df1['value'][i][key][j][0]),math.floor(df1['value'][i][key][j][1])))
            else:
                polygonlabel.append(df1['value'][i][key][0])
    
    
    #denormalizing the points for damaged_car_parts
    for i in range(0,len(df2)):
        for key in df2['value'][i]:
            if key=='points':
                for j in range(len(df2['value'][i][key])):
                    df2['value'][i][key][j]=denormalize(df2['value'][i][key][j],df2['original_width'][i],df2['original_height'][i])
    
    #Dividing the damaged_car_points inot 2 different arrays(pts,polygonlabel) correspondingly for simplicity.            
    import math
    pts2=[[] for i in range(len(df2))]
    polygonlabel2=[]
    for i in range(0,len(df2)):
        for key in df2['value'][i]:
            if key=='points':
                for j in range(len(df2['value'][i][key])):
                    pts2[i].append((math.floor(df2['value'][i][key][j][0]),math.floor(df2['value'][i][key][j][1])))
            else:
                polygonlabel2.append(df2['value'][i][key][0])
    
    #Filling the polygons in car_parts points and making transparencies of the polygon 0.25 by difining the alpha value  
    for i in range(0,len(df1)):
        label=polygonlabel[i]
        array=np.array(pts[i])
        output=img.copy()
        sample_img=img.copy()
    
        if label=='Bumper':
            cv2.fillPoly(sample_img,[array],color=(0,255,0))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(0,255,0),thickness=1,lineType=cv2.LINE_4)
        elif label=='Light':
            cv2.fillPoly(sample_img,[array],color=(255,255,0))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(255,255,0),thickness=1,lineType=cv2.LINE_4)
        elif label=='Windshield':
            cv2.fillPoly(sample_img,[array],color=(0,0,255))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(0,0,255),thickness=1,lineType=cv2.LINE_4)
        elif label=='Mirror':
            cv2.fillPoly(sample_img,[array],color=(255,0,0))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(255,0,0),thickness=1,lineType=cv2.LINE_4)
        elif label=='Boot':
            cv2.fillPoly(sample_img,[array],color=(150,200,100))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(150,200,100),thickness=1,lineType=cv2.LINE_4)
        elif label=='Bonnet':
            cv2.fillPoly(sample_img,[array],color=(100,100,200))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(100,100,200),thickness=1,lineType=cv2.LINE_4)
        elif label=='Fender':
            cv2.fillPoly(sample_img,[array],color=(34,56,76))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(34,56,76),thickness=1,lineType=cv2.LINE_4)
        elif label=='RockerPanel':
            cv2.fillPoly(sample_img,[array],color=(255,127,80))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(255,127,80),thickness=1,lineType=cv2.LINE_4)
        elif label=='Door':
            cv2.fillPoly(sample_img,[array],color=(98,100,45))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(98,100,45),thickness=1,lineType=cv2.LINE_4)
        elif label=='WindowPanel':
            cv2.fillPoly(sample_img,[array],color=(128,0,128))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(128,0,128),thickness=1,lineType=cv2.LINE_4)
        elif label=='NumberPlate':
            cv2.fillPoly(sample_img,[array],color=(128,200,128))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(128,200,128),thickness=1,lineType=cv2.LINE_4)
        elif label=='Radiator':
            cv2.fillPoly(sample_img,[array],color=(128,0,200))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(128,0,200),thickness=1,lineType=cv2.LINE_4)
        else:
            cv2.fillPoly(sample_img,[array],color=(150,150,0))
            cv2.addWeighted(sample_img,0.25,output,0.75,0,output)
            cv2.polylines(output,[array],isClosed=True,color=(150,150,0),thickness=1,lineType=cv2.LINE_4)
        
        img=output.copy()
        
    
        
    #Filling the polygons in damaged_car_parts points and making transparencies of the polygon 0.25 by difining the alpha value 
    for i in range(0,len(df2)):
        label2=polygonlabel2[i]
        array2=np.array(pts2[i])
        output2=img.copy()
        sample_img2=img.copy()
    
        if label2=='Dent&Scratch(zoom)':
            cv2.fillPoly(sample_img2,[array2],color=(89,76,234))
            cv2.addWeighted(sample_img2,0.15,output2,0.85,0,output2)
            cv2.polylines(output2,[array2],isClosed=True,color=(89,76,234),thickness=1,lineType=cv2.LINE_4)

        elif label2=='Broken':
            cv2.fillPoly(sample_img2,[array2],color=(103,45,90))
            cv2.addWeighted(sample_img2,0.15,output2,0.85,0,output2)
            cv2.polylines(output2,[array2],isClosed=True,color=(103,45,90),thickness=1,lineType=cv2.LINE_4)
            
        else:
            cv2.fillPoly(sample_img2,[array2],color=(0,0,0))
            cv2.addWeighted(sample_img2,0.15,output2,0.85,0,output2)
            cv2.polylines(output2,[array2],isClosed=True,color=(0,0,0.15),thickness=1,lineType=cv2.LINE_4)
            
        img=output2.copy()
    result.append(img)
    
    #Function for creating rectangle with label over polygons in car_parts
    def makeRectwithLabel(img,pts,label):
        
        array=np.array(pts)
        mask=np.zeros((img.shape[0],img.shape[1]))
        cv2.fillConvexPoly(mask,array,1)
        mask=mask.astype(bool)
        out1=np.zeros_like(img)
        out1[mask]=img[mask]
        
        #Defining min and max positions for creating rectangle as per the polygons
        x_max,x_min,y_max,y_min,curr_x,curr_y=0,1300,0,1300,0,0
        for i in range(len(pts)):
            curr_x=pts[i][0]
            curr_y=pts[i][1]
            x_max=max(x_max,curr_x)
            x_min=min(x_min,curr_x)
            y_max=max(y_max,curr_y)
            y_min=min(y_min,curr_y)
        (width,height),_=cv2.getTextSize(label,cv2.FONT_ITALIC,0.4,1)
        
        #Identifying different parts in car 
        if label=='Bumper':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(0,255,0),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(0,255,0),1)
        elif label=='Light':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(255,255,0),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(255,255,0),1)
        elif label=='Windshield':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(0,0,255),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(0,0,255),1)
        elif label=='Mirror':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(255,0,0),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(255,0,0),1)
        elif label=='Boot':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(150,200,100),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(150,200,100),1)
        elif label=='Bonnet':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(100,100,200),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(100,100,200),1)
        elif label=='Fender':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(34,56,76),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(34,56,76),1)
        elif label=='RockerPanel':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(255,127,80),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(255,127,80),1)
        elif label=='Door':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(98,100,45),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(98,100,45),1)
        elif label=='WindowPanel':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(128,0,128),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(128,0,128),1)
        elif label=='NumberPlate':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(128,200,128),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(128,200,128),1)
        elif label=='Radiator':
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(128,0,200),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(128,0,200),1)
        else: #wheel
            out1=cv2.rectangle(out1,(x_min,y_min),(x_max,y_max),(150,150,0),1)
            points=[[(x_min,y_min),(x_min,y_min+height+_),(x_min+width,y_min+height+_),(x_min+width,y_min)]]
            points=np.array(points)
            out1=cv2.fillPoly(out1,points,color=(255,255,255))
            cv2.putText(out1,label,(x_min,y_min+10),cv2.FONT_ITALIC,0.4,(150,150,0),1)
            
        
        final=cv2.bitwise_or(img,out1)
        return final
    
    #Function for creating rectangle with label over polygons in damaged_car_parts
    def makeRectwithLabel_damaged_parts(img,pts2,label2):
        
        array2=np.array(pts2)
        mask2=np.zeros((img.shape[0],img.shape[1]))
        cv2.fillConvexPoly(mask2,array2,1)
        mask2=mask2.astype(bool)
        out2=np.zeros_like(img)
        out2[mask2]=img[mask2]
        
        #Defining min and max positions for creating rectangle as per the polygons
        x_max2,x_min2,y_max2,y_min2,curr_x2,curr_y2=0,10,0,10,0,0
        for i in range(len(pts2)):
            curr_x2=pts2[i][0]
            curr_y2=pts2[i][1]
            x_max2=max(x_max2,curr_x2)
            x_min2=min(x_min2,curr_x2)
            y_max2=max(y_max2,curr_y2)
            y_min2=min(y_min2,curr_y2)
        (width2,height2),_=cv2.getTextSize(label2,cv2.FONT_ITALIC,0.4,1)
        
        #Identifying different damaged parts on the car 
        if label2=='Dent&Scratch(zoom)':
            out2=cv2.rectangle(out2,(x_min2,y_min2),(x_max2,y_max2),(89,76,234),1)
            points=[[(x_min2,y_min2),(x_min2,y_min2+height2+_),(x_min2+width2,y_min2+height2+_),(x_min2+width2,y_min2)]]
            points=np.array(points)
            out2=cv2.fillPoly(out2,points,color=(255,255,255))
            cv2.putText(out2,label2,(x_min2,y_min2+10),cv2.FONT_ITALIC,0.4,(89,76,234),1)
        elif label2=='Broken':
            out2=cv2.rectangle(out2,(x_min2,y_min2),(x_max2,y_max2),(103,45,90),1)
            points=[[(x_min2,y_min2),(x_min2,y_min2+height2+_),(x_min2+width2,y_min2+height2+_),(x_min2+width2,y_min2)]]
            points=np.array(points)
            out2=cv2.fillPoly(out2,points,color=(255,255,255))
            cv2.putText(out2,label2,(x_min2,y_min2+10),cv2.FONT_ITALIC,0.4,(103,45,90),1)
            
        else: #Dent&Scratch
            out2=cv2.rectangle(out2,(x_min2,y_min2),(x_max2,y_max2),(90,50,46),1)
            points=[[(x_min2,y_min2),(x_min2,y_min2+height2+_),(x_min2+width2,y_min2+height2+_),(x_min2+width2,y_min2)]]
            points=np.array(points)
            out2=cv2.fillPoly(out2,points,color=(255,255,255))
            cv2.putText(out2,label2,(x_min2,y_min2+10),cv2.FONT_ITALIC,0.4,(90,50,46),1)
            
        final2=cv2.bitwise_or(img,out2)
        return final2
    

    #Creating Rectangle along with label over the corresponding polygons in car_parts
    for i in range(len(df1)):
        output=img.copy()
        arr=np.array(pts[i])
        out1=makeRectwithLabel(output,arr,polygonlabel[i])
        img=out1.copy()
    result.append(img)
    
    #Creating Rectangle along with label over the corresponding polygons in damaged_car_parts
    for i in range(len(df2)):
        output2=img.copy()
        arr2=np.array(pts2[i])
        out2=makeRectwithLabel_damaged_parts(output2,arr2,polygonlabel2[i])
        img=out2.copy()
    result.append(img)
    
    return result


    ##### Function for calculating poygon's area 
    def polygonArea(X, Y):
        
        n=len(X)
        area = 0.0
        j = n - 1
        for i in range(0,n):
            area += (X[j] + X[i]) * (Y[j] - Y[i])
            j = i  
        # Return absolute value
        return int(abs(area / 2.0))  


    def calculate_area(pts,pts2):
        calculated_percentage_of_damage=[]                            
        # Extracting x_coordinates and y_coordinates from points in car_parts 
        
        
        arr1=[]
        arr2=[]
        elements_x=[]
        elements_y=[]
        for i in range(len(pts)): #7
            for j in range(len(pts[i])): #17
                elements_x.append(pts[i][j][0])
                elements_y.append(pts[i][j][1])
            arr1.append([elements_x])
            arr2.append([elements_y])
        
        #Exctracting size of each part's x_coordinates and y_coordinates for car_parts
        sizes=[]
        for i in range(len(pts)):
            sizes.append(len(pts[i]))
            
        #Seperating x_coordinates into 2-D array for car_parts
        it = iter(elements_x)
        splited_x=[[next(it) for i in range(size)] for size in sizes]
        
        
        #Seperating y_coordinates into 2-D array for car_parts
        it = iter(elements_y)
        splited_y=[[next(it) for i in range(size)] for size in sizes]
        

        # Extracting x_coordinates and y_coordinates from points in damaged_car_parts   
        d_arr1=[]
        d_arr2=[]
        d_elements_x=[]
        d_elements_y=[]
        for i in range(len(pts2)): #7
            for j in range(len(pts2[i])): #17
                d_elements_x.append(pts2[i][j][0])
                d_elements_y.append(pts2[i][j][1])
            arr1.append([d_elements_x])
            arr2.append([d_elements_y])
        
        
        #Exctracting size of each part's x_coordinates and y_coordinates for damaged_car_parts
        d_sizes=[]
        for i in range(len(pts2)):
            d_sizes.append(len(pts2[i]))
            
        #Seperating x_coordinates into 2-D array for damaged_car_parts
        it = iter(d_elements_x)
        d_splited_x=[[next(it) for i in range(size)] for size in d_sizes]
        
        #Seperating y_coordinates into 2-D array for damaged_car_parts
        it = iter(d_elements_y)
        d_splited_y=[[next(it) for i in range(size)] for size in d_sizes]
        
        
        #Sum for all x and y coordinates for car_parts 
        arr_sum=[]
        x=0
        y=0
        while x<len(splited_x) and y<len(splited_y):
            area=polygonArea(splited_x[x],splited_y[y])
            arr_sum.append(area)
            x=x+1
            y=y+1

        #Sum for all x and y coordinates for damaged_car_parts 
        d_arr_sum=[]
        x=0
        y=0
        while x<len(d_splited_x) and y<len(d_splited_y):
            area=polygonArea(d_splited_x[x],d_splited_y[y])
            d_arr_sum.append(area)
            x=x+1
            y=y+1

        #Total area of car covered with polygons in 
        Total_area_of_car=sum(arr_sum)
        
        #Total area of damaged_part in the car 
        Total_area_damaged_part=sum(d_arr_sum)
        
        
        #Then you have to find the percentage of the car part damaged (damaged_area / car_part_area) * 100
        percentage_of_car_part_damaged=(Total_area_damaged_part/Total_area_of_car)*100
        
        calculated_percentage_of_damage.append(percentage_of_car_part_damaged)
        return (calculated_percentage_of_damage)

    
    
    