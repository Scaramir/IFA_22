# -*- coding: utf-8 -*-
import os
import shutil

# delete folders ending in specified strings
def delete_folders(path, delete_list):
    subdirs = [x[0] for x in os.walk(path)]
    for folder in subdirs:
        if os.path.basename(os.path.normpath(folder)) in delete_list:
            print('deleting folder: ' + os.path.basename(folder))
            # try with print only first ->
            # should give you e.g. 'deleting folder: 40X'
            
            # then uncomment following line and run again
            #shutil.rmtree(folder)

# ------------------ main ------------------
# main function
if __name__ == '__main__':
    # path to parent folder BreaKHis_v1
    data_path = r'C:\Users\flo17\Documents\FocusAreas\resources\BreaKHis_v1'
    delete_folders(data_path, ["40X", "100X", "400X"])
    