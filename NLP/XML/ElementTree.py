import xml.etree.ElementInclude as ET
import os
import sys

# 遍历xml文件
def traverseXml(element):
    # print (len(element))
    if len(element) > 0:
        for child in element:
            print(child.tag, "----", child.attrib)
            traverseXml(child)

if __name__ == "__main__":
    xmlFilePath = os.path.abspath("F:/Github/DL-data-processing-methods/XML/test.xml")
    print(xmlFilePath)
    try:
        tree = ET.parse(xmlFilePath)
        print("tree type:", type(tree))

        # 获得根节点
        root = tree.getroot()
    except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有异常
        print("parse test.xml fail!")
        sys.exit()
    print("root type:", type(root))
    print(root.tag, "----", root.attrib)

    # 遍历root的下一层
    for child in root:
        print("遍历root的下一层", child.tag, "----", child.attrib)

    # 使用下标访问
    print(root[0].text)
    print(root[1][1][0].text)

    print(20 * "*")
    # 遍历xml文件
    traverseXml(root)
    print(20 * "*")

    # 根据标签名查找root下的所有标签
    captionList = root.findall("item")  # 在当前指定目录下遍历
    print(len(captionList))
    for caption in captionList:
        print(caption.tag, "----", caption.attrib, "----", caption.text)

    # 修改xml文件，将passwd修改为999999
    login = root.find("login")
    passwdValue = login.get("passwd")
    print("not modify passwd:", passwdValue)
    login.set("passwd", "999999")  # 修改，若修改text则表示为login.text
    print("modify passwd:", login.get("passwd"))

