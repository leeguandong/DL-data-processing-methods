import xml.etree.ElementTree as ET

tree = ET.ElementTree(file='0022.xml')

# root = tree.getroot()

# for elem in tree.iter(tag='GeoNE'):
#     print(elem.tag, elem.attrib)

for elem in tree.iterfind('AnnotationSet/Annotation[@Type="GeoNE"]'):
    print(elem.tag, elem.attrib)
