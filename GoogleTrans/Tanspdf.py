'''
https://blog.csdn.net/cheany/article/details/79109903
'''

from PyPDF2.pdf import PdfFileReader, PdfFileWriter, ContentStream


# 读取 PDF 文件的代码，这是获取所有页的文本
def getDataUsingPyPdf2(filename):
    pdf = PdfFileReader(open(filename, "rb"))
    content = ""
    num = pdf.getNumPages()
    for i in range(0, num):
        extractedText = pdf.getPage(i).extractText()
        content += extractedText + "\n"
    return content


# 对每页的文本进行翻译处理
'''
这个函数其实是从extractText摘取出来的，只是为了更灵活的对文本进行处理而已，
因为PDF的文本是割裂的，需要拼接起来，这个其实没完全处理好，只是做了简单的处理。
'''


def dopage(page):
    content = page["/Contents"].getObject()
    if not isinstance(content, ContentStream):
        content = ContentStream(content, pdf)

    text = u_("")
    for operands, operator in content.operations:
        # print operator, operands
        if operator == b_("Tj"):
            _text = operands[0]
            if isinstance(_text, TextStringObject):
                text += _text + " "
        elif operator == b_("rg"):
            text += "\n"
        elif operator == b_("T*"):
            text += "\n"
        elif operator == b_("'"):
            text += "\n"
            _text = operands[0]
            if isinstance(_text, TextStringObject):
                text += operands[0] + " "
        elif operator == b_('"'):
            _text = operands[2]
            if isinstance(_text, TextStringObject):
                text += _text + " "
        elif operator == b_("TJ"):
            for i in operands[0]:
                if isinstance(i, TextStringObject):
                    text += i
            text += " "

    texts = text.split('. ')
    results = ''
    for i in range(len(texts)):
        try:
            results = results + translate(str(texts[i])) + "\n"
        except Exception as e:
            print
            e
    return results


import re
import execjs
import urllib, urllib2
import sys
import json


class Py4Js():
    def __init__(self):
        self.ctx = execjs.compile("""
        function TL(a) {
        var k = "";
        var b = 406644;
        var b1 = 3293161072;

        var jd = ".";
        var $b = "+-a^+6";
        var Zb = "+-3^+b+-f";

        for (var e = [], f = 0, g = 0; g < a.length; g++) {
            var m = a.charCodeAt(g);
            128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023),
            e[f++] = m >> 18 | 240,
            e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224,
            e[f++] = m >> 6 & 63 | 128),
            e[f++] = m & 63 | 128)
        }
        a = b;
        for (f = 0; f < e.length; f++) a += e[f],
        a = RL(a, $b);
        a = RL(a, Zb);
        a ^= b1 || 0;
        0 > a && (a = (a & 2147483647) + 2147483648);
        a %= 1E6;
        return a.toString() + jd + (a ^ b)
    };

    function RL(a, b) {
        var t = "a";
        var Yb = "+";
        for (var c = 0; c < b.length - 2; c += 3) {
            var d = b.charAt(c + 2),
            d = d >= t ? d.charCodeAt(0) - 87 : Number(d),
            d = b.charAt(c + 1) == Yb ? a >>> d: a << d;
            a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d
        }
        return a
    }
    """)

    def getTk(self, text):
        return self.ctx.call("TL", text)


def open_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib2.Request(url=url, headers=headers)
    response = urllib2.urlopen(req)
    data = response.read().decode('utf-8')
    return data


def translate(content):
    # print "content: ", content
    js = Py4Js()
    tk = js.getTk(content)

    texts = ""
    content = urllib2.quote(content)
    url = "http://translate.google.cn/translate_a/single?client=t" \
          "&sl=EN&tl=zh-CN&hl=zh-CNdt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca" \
          "&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1" \
          "&srcrom=0&ssel=0&tsel=0&kc=2&tk=%s&q=%s" % (tk, content)

    result = open_url(url)
    re = json.loads(result)
    str = ""
    for i in re[0]:
        if i[0]:
            str += i[0]
            # print " ========>", i[0]
    return str


if __name__ == "__main__":
    text = "您好"
    texts = text.split('.')
    results = ''
    for i in range(len(texts)):
        try:
            results = results + translate(str(texts[i]))
        except Exception as e:
            print(e)
    print(results)
