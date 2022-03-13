1、

sys添加指定目錄為source root的代碼方法：

```python
import sys
sys.path.append('/content/drive/MyDrive/visual_motif_removal')
sys.path.insert(0, '/content/drive/MyDrive/visual_motif_removal')
```

2、

os.walk(path) 

os.path.splitext()

os.path.split()

os.path.join()

```python
"""
root:所指的是当前正在遍历的这个文件夹的本身的地址,即path
dirs:是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
files:同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
"""
for root, _, files in os.walk(_source):
    for file in files:
        """
        os.path.splitext:分割路径，返回路径名和文件扩展名的元组
        os.path.split:把路径分割成 dirname 和 basename，返回一个元组(以最後一個分隔符為界限)
        """
        file_name, file_extension = os.path.splitext(file)
        """
        os.path.join(path1, path2,..., path):用分隔符將所有path鏈接
        """
        paths.append(os.path.join(root, file))
return paths
```

3、shutil

複製文件

```python
shutil.copy(os.path.join(root, file), os.path.join(des, prefix+".jpg"))
```

4、

