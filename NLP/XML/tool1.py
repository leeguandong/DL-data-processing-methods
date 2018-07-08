f = open('train_ner_loc1', encoding='utf-8')
# total_content = f.read()
# length = len(total_content)
# print(length)
# # 8632
# 316
# 661
# 311
# 256
# 517

# import xml.etree.ElementTree as ET
#
# tree = ET.ElementTree(file='0025.xml')

# root = tree.getroot()

# for elem in tree.iter(tag='GeoNE'):
#     print(elem.tag, elem.attrib)

# for elem in tree.iterfind('AnnotationSet/Annotation[@Type="GeoNE"]'):
#     print(elem.tag, elem.attrib)

# print(tree.find('AnnotationSet/Annotation[@Type="GeoNE"]').tag)

for i in range(200000):
    content = f.readline()
    # print(content + '  O  ' + str(i))
    if '。' in content:
        print(content)
    else:
        print(content, end='')
