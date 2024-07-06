import icedata
import os
import shutil
import glob
data_url = 'https://s3.amazonaws.com/fast-ai-sample/camvid_tiny.tgz'
data = icedata.load_data(data_url,"data")


destination_dir = os.getcwd()  # ใช้ไดเรกทอรีปัจจุบันในการทำงาน

# เคลื่อนย้ายโฟลเดอร์ไปยังไดเรกทอรีปลายทาง
shutil.move(str(data), os.path.join(destination_dir, 'data'))

folder_path = "data/camvid_tiny/labels"  # เปลี่ยนเป็น path ของโฟลเดอร์ของคุณ

for filename in os.listdir(folder_path):
    if filename.endswith("_P.png"):
        new_filename = filename.replace("_P", "") 
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print(data)