import shutil
import os

for file_name in os.listdir("original_pictures"):
    if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
        #증명사진/취업사진을 판단하는 함수를 밑에 적는다
        temp = detector.generate("csimg\\" + file_name, False)
        if temp == "jmsg":
            #여자남자를 판단하는 함수를 밑에 적는다
            temp_2 = detector.generate("csimg\\" + file_name, False)
            
            if temp_2 == "woman":
                shutil.move("original_pictures\\"+file_name, "jmsg\\womans\\"+file_name)
            else:
                shutil.move("original_pictures\\"+file_name, "jmsg\\mans\\"+file_name)

        else:
            #여자남자를 판단하는 함수를 밑에 적는다
            temp_2 = detector.generate("csimg\\" + file_name, False)

            if temp_2 == "woman":
                shutil.move("original_pictures\\"+file_name, "chupsg\\womans\\"+file_name)
            else:
                shutil.move("original_pictures\\"+file_name, "chupsg\\mans\\"+file_name)
