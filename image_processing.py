import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, datetime, timedelta
import pandas as pd
from PIL import Image
from IPython.display import display
import argparse
import time




def run(source, destination, model_path='./best_new_model.pt'):
    print('#################################################################')
    print('Source: '+str(source))
    print('Reports: '+str(destination))
    print('ML Model: '+str(model_path))
    print('#################################################################')
    s3_file_format = '%Y%m%d%H%M%S'
    date_format = "%Y-%m-%d"

    current_file_date = datetime.now().strftime(date_format)
    report_file_date = datetime.now().strftime(s3_file_format)
    s_no = 0
    # number of data points in the data
    path=source
    files=os.listdir(path)
    #print("Number of Data Points:" , len(files))

    out_df = pd.DataFrame(columns=['S.no.','Date','Filename','% of Rack Vacany', 'No of Racks Detected'])


    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)


    for file in files:
        s_no +=1
        file_name = path+str(file)
        img = file_name
        print('Image Loaded :'+str(img))
        # Inference
        results = model(img)
        print('Cabinet Detection Results:')
        print(results)
        results.save()
        df = results.pandas().xyxy[0]
        print(img)
        im = Image.open(img)
        display(im)
        #percent=list()
        percent_170=list()
        #percent_180=list()
        #percent_190=list()
        #percent_200=list()
        #percent_210=list()
        #percent_220=list()
        #percent_230=list()
        percent_2=list()
        for index, row in df.iterrows():
            #print(index)
            #print(df['xmin'][index])
            #print(df['ymin'][index])
            #print(df['xmax'][index])
            #print(df['ymax'][index])
            #im1 = im.crop( (df['xmin'], df['ymin'], df['xmax'], df['ymax']) )
            im_ = im.crop((df['xmin'][index], df['ymin'][index], df['xmax'][index], df['ymax'][index]))
            #display(im_)
            open_cv_image = np.array(im_) 
            
            #170 Algo
            grayImage = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
            (thresh, blackAndWhiteImagep170) = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)
            #(thresh, blackAndWhiteImagep180) = cv2.threshold(grayImage, 180, 255, cv2.THRESH_BINARY)
            #(thresh, blackAndWhiteImagep190) = cv2.threshold(grayImage, 190, 255, cv2.THRESH_BINARY)
            #(thresh, blackAndWhiteImagep200) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
            #(thresh, blackAndWhiteImagep210) = cv2.threshold(grayImage, 210, 255, cv2.THRESH_BINARY)
            #(thresh, blackAndWhiteImagep220) = cv2.threshold(grayImage, 220, 255, cv2.THRESH_BINARY)
            #(thresh, blackAndWhiteImagep230) = cv2.threshold(grayImage, 230, 255, cv2.THRESH_BINARY)

                
            #cv2.imshow('Black white image', blackAndWhiteImage)


            #cv2.imshow('Original image',originalImage)
            #cv2.imshow('Gray image', grayImage)
            #print(blackAndWhiteImage.size)
            
            n_white_pix170 = np.sum(blackAndWhiteImagep170 == 255)
            #n_white_pix180 = np.sum(blackAndWhiteImagep180 == 255)
            #n_white_pix190 = np.sum(blackAndWhiteImagep190 == 255)
            #n_white_pix200 = np.sum(blackAndWhiteImagep200 == 255)
            #n_white_pix210 = np.sum(blackAndWhiteImagep210 == 255)
            #n_white_pix220 = np.sum(blackAndWhiteImagep220 == 255)
            #n_white_pix230 = np.sum(blackAndWhiteImagep230 == 255)
            
            #print('Number of white pixels:', n_white_pix)
            percent170 = (n_white_pix170/blackAndWhiteImagep170.size)*100
            #percent180 = (n_white_pix180/blackAndWhiteImagep180.size)*100
            #percent190 = (n_white_pix190/blackAndWhiteImagep190.size)*100
            #percent200 = (n_white_pix200/blackAndWhiteImagep200.size)*100
            #percent210 = (n_white_pix210/blackAndWhiteImagep210.size)*100
            #percent220 = (n_white_pix220/blackAndWhiteImagep220.size)*100
            #percent230 = (n_white_pix230/blackAndWhiteImagep230.size)*100
            
            percent_170.append(percent170)
            #percent_180.append(percent180)
            #percent_190.append(percent190)
            #percent_200.append(percent200)
            #percent_210.append(percent210)
            #percent_220.append(percent220)
            #percent_230.append(percent230)
            
            
            # Ashwath algo
            grid_RGB = cv2.cvtColor(open_cv_image,cv2.COLOR_BGR2RGB)
            grid_HSV = cv2.cvtColor(grid_RGB,cv2.COLOR_RGB2HSV)
            upper = np.array([180, 18, 230])
            lower = np.array([0, 0, 40])

            mask = cv2.inRange(grid_HSV,lower,upper)
            res = cv2.bitwise_and(grid_RGB,grid_RGB,mask = mask)

            ratio_white = cv2.countNonZero(mask)/(open_cv_image.size/3)
            colorPercent = (ratio_white * 100)
            percent_2.append(colorPercent)
            #print('White pixel percentage: ' + str(np.round(colorPercent, 2)) +' %')
            #print(percent)
        try:
            avg_170 = sum(percent_170)/len(percent_170)
            #avg_180 = sum(percent_180)/len(percent_180)
            #avg_190 = sum(percent_190)/len(percent_190)
            #avg_200 = sum(percent_200)/len(percent_200)
            #avg_210 = sum(percent_210)/len(percent_210)
            #avg_220 = sum(percent_220)/len(percent_220)
            #avg_230 = sum(percent_230)/len(percent_230)
            avg_2 = sum(percent_2)/len(percent_2)
        except ZeroDivisionError:
            avg_170=0
            #avg_180=0
            #avg_190=0
            #avg_200=0
            #avg_210=0
            #avg_220=0
            #avg_230=0
            avg_2=0
            
            #print('No Cabinets Found')
        out_170 = round(avg_170,2)
        #out_180 = round(avg_180,2)
        #out_190 = round(avg_190,2)
        #out_200 = round(avg_200,2)
        #out_210 = round(avg_210,2)
        #out_220 = round(avg_220,2)
        #out_230 = round(avg_230,2)
        out_2 = round(avg_2,2)
        total_percent=[out_170, out_2]
        avg_percent = round(sum(total_percent)/len(total_percent),2)
        #new_row = {'S.no.':s_no, 'Date':current_file_date, 'Filename':file, 'Rack Vacany Percentage':out, 'Racks No': len(percent)}
        df_new_row = pd.DataFrame({'S.no.':[s_no], 'Date':[current_file_date], 'Filename':[file], '% of Rack Vacany': [avg_percent], 'No of Racks Detected': [len(percent_2)]})
        #out_df = out_df.append(new_row, ignore_index=True)
        out_df = pd.concat([out_df, df_new_row])


    out_df.to_csv(destination+'/rack_vacany_report_new_model'+report_file_date+'.csv',index=False)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default= '02_testing_images/testing89/')
    parser.add_argument('--destination', type=str, default= '04_output_reports/')
    parser.add_argument('--model_path', type=str, default= './best_new_model.pt')
    opt = parser.parse_args()
    #print(vars(opt))
    return opt


def main(opt):
    start = time.time()
    run(**vars(opt))
    time_taken=time.time()-start
    print('Total run time: '+str(round(time_taken,2))+' seconds')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)