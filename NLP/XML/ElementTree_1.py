import xml.etree.ElementTree as ET

#################################################
# 1.将XML文档解析成树(tree)
tree = ET.ElementTree(file='doc.xml')

# 获取根元素,根元素root是一个Element对象
root = tree.getroot()

# root拥有标签和属性字典
print(root.tag, root.attrib)
# doc {}

# 根元素并没有属性，与其他element对象一样，根元素也具备遍历其直接子元素的接口
for child in root:
    print(child.tag, child.attrib)
# branch {'hash': '1cdf045c', 'name': 'codingpy.com'}
# branch {'hash': 'f200013e', 'name': 'release01'}
# branch {'name': 'invalid'}

# 通过索引值来访问特定的子元素
print(root[0].tag)
print(root[0].text)
# root[0]是一个elementTree元素
print('#' * 30, root[0])
# branch
# text,source

###############################################
# 2.查找需要的元素
# 可以发现我们能够通过递归方法获取树中所有元素。但是elementTree封装了一些方法
# Element对象有一个iter方法，可以对某个元素对象之下所有的子元素进行深度优先遍历。
for elem in tree.iter():
    print(elem.tag, elem.attrib)
# branch {'hash': '1cdf045c', 'name': 'codingpy.com'}
# branch {'hash': 'f200013e', 'name': 'release01'}
# sub-branch {'name': 'subrelease01'}
# branch {'name': 'invalid'}

# 对树进行任意遍历-遍历所有元素
for elem in tree.iter(tag='branch'):
    print(elem.tag, elem.attrib, elem.text)

#########
# 3.支持通过Xpath查找元素
# elementTree对象中有一些find方法可以接受Xpath路径作为参数，find会返回第一个匹配的子元素，findall以列表的形式
# 返回所有匹配的子元素，iterfind则会返回一个所有匹配元素的迭代器。ElementTree对象也具备这些方法，相应的查找是从根节点开始的
for elem in tree.iterfind('branch/sub-branch'):
    print(elem.tag, elem.attrib)

# 具备某个name属性的branch
for elem in tree.iterfind('branch[@name="release01"]'):
    print(elem.tag,elem.attrib)














