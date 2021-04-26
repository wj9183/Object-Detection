import streamlit as st
from menu.object_detection import object_detection
from menu.ssd import ssd
from menu.yolo import yolo
from menu.semantic_segmentation import semantic_segmentation

def main():
    menu = ['Object Detection', 'SSD', 'YOLO', 'Semantic Segmentation']
    choice = st.sidebar.selectbox("메뉴", menu)
    
    
    if choice == 'Object Detection':
        object_detection()
    
    elif choice == 'SSD':
        ssd()

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