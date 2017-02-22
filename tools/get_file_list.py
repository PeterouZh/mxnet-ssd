import os

def main():
    GetFileListOrder('.')

def GetFileListOrder(file_path, endswith_flag = False, end_strs = None,
                     label_flag = False, label = 0):
    """
     Walk a directory recursively and return the file lists in order
    
    Parameters:
    ----------
     file_path : str
         directory to look through	  
     endswith_flag : bool
         find specific file ends with end_str
     end_strs : list of str
         specify the end string
     label_flag : bool
         whether add label after file path
     label : int
         value of label
    Returns:
    ----------
     file_list : list
    """
    for parent, dirnames, filenames in os.walk(file_path, topdown = True): 
        dirnames.sort(cmp = lambda x,y: cmp(x.lower(), y.lower()))
        filenames.sort(cmp = lambda x,y: cmp(x.lower(), y.lower()))
        
        return_str = []   
        for dirname in  dirnames:
            sub_dir = os.path.join(parent, dirname)
            return_str += GetFileListOrder(sub_dir, endswith_flag, end_strs, label_flag, label)
            label += 1
            
        for filename in filenames:
            file_name = os.path.join(parent,filename)
            if endswith_flag:
                if not isinstance(end_strs, list):
                    end_strs = [end_strs]
                for end_str in end_strs:
                    if file_name.endswith(end_str):
                        if label_flag:
                            return_str.append(file_name + ' %d'%label)
                        else:
                            return_str.append(file_name)
                        break
                continue
            else:
                if label_flag:
                    return_str.append(file_name + ' %d'%label)
                else:
                    return_str.append(file_name)
        break          # os.walk() will look through file_path recursively
    return return_str
    
if __name__ == '__main__':
    main()
