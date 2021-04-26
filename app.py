import streamlit as st
from menu.object_detection import object_detection
from menu.ssd import ssd_image, ssd_video
from menu.yolo import yolo
from menu.semantic_segmentation import semantic_segmentation

def main():
    menu = ['Object Detection', 'SSD', 'YOLO', 'Semantic Segmentation']
    choice = st.sidebar.selectbox("메뉴", menu)
    
    
    if choice == 'Object Detection':
        object_detection()
    
    elif choice == 'SSD':
        menu = ['Object Detection', '이미지', '동영상']
        choice = st.sidebar.selectbox("메뉴", menu)
        if choice == '이미지':
            ssd_image()
        elif choice == '동영상':
            ssd_video()

    elif choice == 'YOLO':
        yolo()
    
    elif choice == 'Semantic Segmentation':
        semantic_segmentation()
    
    else:
        #메뉴를 잘못 선택한 경우 에러코드를 반환합니다.
        {"err_code" : 0}
        pass


if __name__ == '__main__':
    main()